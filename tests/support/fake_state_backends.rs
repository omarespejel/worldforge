use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use worldforge_core::error::{Result, WorldForgeError};
use worldforge_core::state::S3Config;

pub fn test_s3_config(endpoint: &str) -> S3Config {
    S3Config {
        bucket: "worldforge-tests".to_string(),
        region: "us-east-1".to_string(),
        endpoint: Some(endpoint.to_string()),
        access_key_id: "test-access".to_string(),
        secret_access_key: "test-secret".to_string(),
        session_token: Some("test-session".to_string()),
        prefix: "states".to_string(),
    }
}

#[derive(Debug, Clone)]
pub struct RecordedS3Request {
    pub method: String,
    pub path: String,
    pub query: String,
    pub headers: HashMap<String, String>,
}

pub struct FakeS3Server {
    endpoint: String,
    pub requests: Arc<Mutex<Vec<RecordedS3Request>>>,
    handle: JoinHandle<()>,
}

impl FakeS3Server {
    pub async fn spawn() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let endpoint = format!("http://{address}");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let objects = Arc::new(Mutex::new(HashMap::new()));
        let requests_for_task = Arc::clone(&requests);
        let objects_for_task = Arc::clone(&objects);

        let handle = tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(pair) => pair,
                    Err(_) => break,
                };
                let requests = Arc::clone(&requests_for_task);
                let objects = Arc::clone(&objects_for_task);
                tokio::spawn(async move {
                    let _ = handle_fake_s3_connection(stream, requests, objects).await;
                });
            }
        });

        Self {
            endpoint,
            requests,
            handle,
        }
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

impl Drop for FakeS3Server {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

pub struct FakeRedisServer {
    pub address: SocketAddr,
    pub commands: Arc<Mutex<Vec<Vec<String>>>>,
    handle: JoinHandle<()>,
}

impl FakeRedisServer {
    pub async fn spawn() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let commands = Arc::new(Mutex::new(Vec::new()));
        let values = Arc::new(Mutex::new(HashMap::new()));
        let sets = Arc::new(Mutex::new(HashMap::new()));
        let commands_for_task = Arc::clone(&commands);
        let values_for_task = Arc::clone(&values);
        let sets_for_task = Arc::clone(&sets);

        let handle = tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(pair) => pair,
                    Err(_) => break,
                };
                let commands = Arc::clone(&commands_for_task);
                let values = Arc::clone(&values_for_task);
                let sets = Arc::clone(&sets_for_task);
                tokio::spawn(async move {
                    let _ = handle_fake_redis_connection(stream, commands, values, sets).await;
                });
            }
        });

        Self {
            address,
            commands,
            handle,
        }
    }

    pub fn url(&self, database: u32) -> String {
        format!("redis://{}/{database}", self.address)
    }
}

impl Drop for FakeRedisServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

async fn handle_fake_s3_connection(
    stream: TcpStream,
    requests: Arc<Mutex<Vec<RecordedS3Request>>>,
    objects: Arc<Mutex<HashMap<String, Vec<u8>>>>,
) -> Result<()> {
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    if reader
        .read_line(&mut request_line)
        .await
        .map_err(|error| WorldForgeError::InternalError(error.to_string()))?
        == 0
    {
        return Ok(());
    }

    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| WorldForgeError::InvalidState("missing fake s3 method".to_string()))?
        .to_string();
    let target = parts
        .next()
        .ok_or_else(|| WorldForgeError::InvalidState("missing fake s3 target".to_string()))?
        .to_string();

    let mut headers = HashMap::new();
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        let read = reader
            .read_line(&mut line)
            .await
            .map_err(|error| WorldForgeError::InternalError(error.to_string()))?;
        if read == 0 || line == "\r\n" {
            break;
        }

        let trimmed = line.trim_end();
        if let Some((name, value)) = trimmed.split_once(':') {
            let key = name.trim().to_ascii_lowercase();
            let value = value.trim().to_string();
            if key == "content-length" {
                content_length = value.parse::<usize>().map_err(|error| {
                    WorldForgeError::InvalidState(format!(
                        "invalid fake s3 content length '{value}': {error}"
                    ))
                })?;
            }
            headers.insert(key, value);
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader
            .read_exact(&mut body)
            .await
            .map_err(|error| WorldForgeError::InternalError(error.to_string()))?;
    }

    let (path, query) = match target.split_once('?') {
        Some((path, query)) => (path.to_string(), query.to_string()),
        None => (target, String::new()),
    };

    requests.lock().await.push(RecordedS3Request {
        method: method.clone(),
        path: path.clone(),
        query: query.clone(),
        headers,
    });

    let mut stream = reader.into_inner();
    let query_params = parse_fake_query(&query);
    if query_params.get("list-type").map(String::as_str) == Some("2") {
        let prefix = query_params.get("prefix").cloned().unwrap_or_default();
        let objects = objects.lock().await;
        let body = build_fake_s3_list_response(&objects, &prefix);
        write_fake_http_response(
            &mut stream,
            200,
            "OK",
            body.as_bytes(),
            Some("application/xml"),
        )
        .await?;
        return Ok(());
    }

    let key = fake_s3_object_key(&path);
    match method.as_str() {
        "PUT" => {
            objects.lock().await.insert(key, body);
            write_fake_http_response(&mut stream, 200, "OK", b"", None).await?;
        }
        "GET" => {
            if let Some(payload) = objects.lock().await.get(&key).cloned() {
                write_fake_http_response(
                    &mut stream,
                    200,
                    "OK",
                    &payload,
                    Some("application/octet-stream"),
                )
                .await?;
            } else {
                write_fake_http_response(&mut stream, 404, "Not Found", b"", None).await?;
            }
        }
        "HEAD" => {
            let status = if objects.lock().await.contains_key(&key) {
                200
            } else {
                404
            };
            let reason = if status == 200 { "OK" } else { "Not Found" };
            write_fake_http_response(&mut stream, status, reason, b"", None).await?;
        }
        "DELETE" => {
            objects.lock().await.remove(&key);
            write_fake_http_response(&mut stream, 204, "No Content", b"", None).await?;
        }
        other => {
            let body = format!("unsupported method: {other}");
            write_fake_http_response(
                &mut stream,
                405,
                "Method Not Allowed",
                body.as_bytes(),
                Some("text/plain"),
            )
            .await?;
        }
    }

    Ok(())
}

fn fake_s3_object_key(path: &str) -> String {
    path.trim_start_matches('/')
        .split_once('/')
        .map(|(_, key)| percent_decode(key))
        .unwrap_or_default()
}

fn parse_fake_query(query: &str) -> HashMap<String, String> {
    query
        .split('&')
        .filter(|pair| !pair.is_empty())
        .map(|pair| match pair.split_once('=') {
            Some((key, value)) => (percent_decode(key), percent_decode(value)),
            None => (percent_decode(pair), String::new()),
        })
        .collect()
}

fn build_fake_s3_list_response(objects: &HashMap<String, Vec<u8>>, prefix: &str) -> String {
    let mut keys = objects
        .keys()
        .filter(|key| key.starts_with(prefix))
        .cloned()
        .collect::<Vec<_>>();
    keys.sort();

    let contents = keys
        .iter()
        .map(|key| format!("<Contents><Key>{key}</Key></Contents>"))
        .collect::<String>();

    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><ListBucketResult><Name>worldforge-tests</Name><Prefix>{prefix}</Prefix><KeyCount>{}</KeyCount><MaxKeys>1000</MaxKeys><IsTruncated>false</IsTruncated>{contents}</ListBucketResult>",
        keys.len()
    )
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut index = 0usize;
    let mut decoded = Vec::with_capacity(bytes.len());
    while index < bytes.len() {
        match bytes[index] {
            b'%' if index + 2 < bytes.len() => {
                if let (Some(high), Some(low)) =
                    (hex_value(bytes[index + 1]), hex_value(bytes[index + 2]))
                {
                    decoded.push((high << 4) | low);
                    index += 3;
                    continue;
                }
                decoded.push(bytes[index]);
                index += 1;
            }
            b'+' => {
                decoded.push(b' ');
                index += 1;
            }
            other => {
                decoded.push(other);
                index += 1;
            }
        }
    }

    String::from_utf8(decoded).unwrap_or_default()
}

async fn write_fake_http_response(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    body: &[u8],
    content_type: Option<&str>,
) -> Result<()> {
    let mut response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    if let Some(content_type) = content_type {
        response.push_str(&format!("Content-Type: {content_type}\r\n"));
    }
    response.push_str("\r\n");
    stream
        .write_all(response.as_bytes())
        .await
        .map_err(|error| WorldForgeError::InternalError(error.to_string()))?;
    stream
        .write_all(body)
        .await
        .map_err(|error| WorldForgeError::InternalError(error.to_string()))?;
    stream
        .flush()
        .await
        .map_err(|error| WorldForgeError::InternalError(error.to_string()))?;
    Ok(())
}

async fn handle_fake_redis_connection(
    stream: TcpStream,
    commands: Arc<Mutex<Vec<Vec<String>>>>,
    values: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    sets: Arc<Mutex<HashMap<String, HashSet<String>>>>,
) -> Result<()> {
    let mut reader = BufReader::new(stream);

    loop {
        let request = match read_redis_value(&mut reader).await {
            Ok(request) => request,
            Err(WorldForgeError::InternalError(message))
                if message.contains("unexpected EOF while reading Redis response") =>
            {
                break;
            }
            Err(error) => return Err(error),
        };

        let RedisValue::Array(items) = request else {
            return Err(WorldForgeError::InvalidState(
                "redis request must be an array".to_string(),
            ));
        };

        let mut command = Vec::with_capacity(items.len());
        for item in items {
            command.push(redis_value_to_string(item)?);
        }
        if command.is_empty() {
            return Err(WorldForgeError::InvalidState(
                "redis request is empty".to_string(),
            ));
        }

        commands.lock().await.push(command.clone());

        let response = match command[0].as_str() {
            "PING" => redis_simple_string_response(reader.get_mut(), "PONG").await,
            "SELECT" => redis_simple_string_response(reader.get_mut(), "OK").await,
            "SET" => {
                if command.len() != 3 {
                    return Err(WorldForgeError::InvalidState(
                        "SET requires key and value".to_string(),
                    ));
                }
                let key = command[1].clone();
                let value = command[2].clone().into_bytes();
                values.lock().await.insert(key, value);
                redis_simple_string_response(reader.get_mut(), "OK").await
            }
            "GET" => {
                if command.len() != 2 {
                    return Err(WorldForgeError::InvalidState(
                        "GET requires key".to_string(),
                    ));
                }
                let key = &command[1];
                let value = values.lock().await.get(key).cloned();
                redis_bulk_response(reader.get_mut(), value.as_deref()).await
            }
            "DEL" => {
                if command.len() != 2 {
                    return Err(WorldForgeError::InvalidState(
                        "DEL requires key".to_string(),
                    ));
                }
                let removed = values.lock().await.remove(&command[1]).is_some();
                redis_integer_response(reader.get_mut(), if removed { 1 } else { 0 }).await
            }
            "SADD" => {
                if command.len() != 3 {
                    return Err(WorldForgeError::InvalidState(
                        "SADD requires key and member".to_string(),
                    ));
                }
                let mut sets = sets.lock().await;
                let members = sets.entry(command[1].clone()).or_default();
                let inserted = members.insert(command[2].clone());
                redis_integer_response(reader.get_mut(), if inserted { 1 } else { 0 }).await
            }
            "SMEMBERS" => {
                if command.len() != 2 {
                    return Err(WorldForgeError::InvalidState(
                        "SMEMBERS requires key".to_string(),
                    ));
                }
                let mut members = sets
                    .lock()
                    .await
                    .get(&command[1])
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .collect::<Vec<_>>();
                members.sort_unstable();
                let members = members
                    .into_iter()
                    .map(|member| member.into_bytes())
                    .collect::<Vec<_>>();
                redis_array_response(reader.get_mut(), &members).await
            }
            "SREM" => {
                if command.len() != 3 {
                    return Err(WorldForgeError::InvalidState(
                        "SREM requires key and member".to_string(),
                    ));
                }
                let mut sets = sets.lock().await;
                let removed = sets
                    .get_mut(&command[1])
                    .map(|members| members.remove(&command[2]))
                    .unwrap_or(false);
                redis_integer_response(reader.get_mut(), if removed { 1 } else { 0 }).await
            }
            other => {
                redis_error_response(reader.get_mut(), &format!("unknown command '{other}'")).await
            }
        };

        response.map_err(|error| {
            WorldForgeError::InternalError(format!("fake redis server write failed: {error}"))
        })?;
    }

    Ok(())
}

#[derive(Debug)]
enum RedisValue {
    SimpleString(String),
    Error(String),
    Integer(i64),
    BulkString(Vec<u8>),
    Array(Vec<RedisValue>),
    Null,
}

fn read_redis_value<'a>(
    reader: &'a mut BufReader<TcpStream>,
) -> Pin<Box<dyn Future<Output = Result<RedisValue>> + Send + 'a>> {
    Box::pin(async move {
        let mut prefix = [0u8; 1];
        reader.read_exact(&mut prefix).await.map_err(|error| {
            WorldForgeError::InternalError(format!("failed to read Redis response: {error}"))
        })?;

        match prefix[0] {
            b'+' => Ok(RedisValue::SimpleString(read_redis_line(reader).await?)),
            b'-' => Ok(RedisValue::Error(read_redis_line(reader).await?)),
            b':' => {
                let line = read_redis_line(reader).await?;
                let value = line.parse::<i64>().map_err(|_| {
                    WorldForgeError::InvalidState(format!(
                        "invalid Redis integer response '{line}'"
                    ))
                })?;
                Ok(RedisValue::Integer(value))
            }
            b'$' => {
                let line = read_redis_line(reader).await?;
                let length = line.parse::<isize>().map_err(|_| {
                    WorldForgeError::InvalidState(format!("invalid Redis bulk length '{line}'"))
                })?;
                if length < 0 {
                    return Ok(RedisValue::Null);
                }
                let mut payload = vec![0u8; length as usize];
                reader.read_exact(&mut payload).await.map_err(|error| {
                    WorldForgeError::InternalError(format!(
                        "failed to read Redis bulk payload: {error}"
                    ))
                })?;
                let mut crlf = [0u8; 2];
                reader.read_exact(&mut crlf).await.map_err(|error| {
                    WorldForgeError::InternalError(format!(
                        "failed to read Redis bulk terminator: {error}"
                    ))
                })?;
                if crlf != *b"\r\n" {
                    return Err(WorldForgeError::InvalidState(
                        "invalid Redis bulk string terminator".to_string(),
                    ));
                }
                Ok(RedisValue::BulkString(payload))
            }
            b'*' => {
                let line = read_redis_line(reader).await?;
                let count = line.parse::<isize>().map_err(|_| {
                    WorldForgeError::InvalidState(format!("invalid Redis array length '{line}'"))
                })?;
                if count < 0 {
                    return Ok(RedisValue::Null);
                }

                let mut values = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    values.push(read_redis_value(reader).await?);
                }
                Ok(RedisValue::Array(values))
            }
            other => Err(WorldForgeError::InvalidState(format!(
                "unsupported Redis response prefix '{}'",
                other as char
            ))),
        }
    })
}

async fn read_redis_line(reader: &mut BufReader<TcpStream>) -> Result<String> {
    let mut line = String::new();
    let bytes = reader.read_line(&mut line).await.map_err(|error| {
        WorldForgeError::InternalError(format!("failed to read Redis line: {error}"))
    })?;

    if bytes == 0 {
        return Err(WorldForgeError::InternalError(
            "unexpected EOF while reading Redis response".to_string(),
        ));
    }

    if line.ends_with('\n') {
        line.pop();
    }
    if line.ends_with('\r') {
        line.pop();
    }

    Ok(line)
}

async fn redis_simple_string_response(stream: &mut TcpStream, value: &str) -> std::io::Result<()> {
    stream.write_all(format!("+{value}\r\n").as_bytes()).await
}

async fn redis_error_response(stream: &mut TcpStream, message: &str) -> std::io::Result<()> {
    stream
        .write_all(format!("-ERR {message}\r\n").as_bytes())
        .await
}

async fn redis_integer_response(stream: &mut TcpStream, value: i64) -> std::io::Result<()> {
    stream.write_all(format!(":{value}\r\n").as_bytes()).await
}

async fn redis_bulk_response(stream: &mut TcpStream, value: Option<&[u8]>) -> std::io::Result<()> {
    match value {
        Some(bytes) => {
            stream
                .write_all(format!("${}\r\n", bytes.len()).as_bytes())
                .await?;
            stream.write_all(bytes).await?;
            stream.write_all(b"\r\n").await
        }
        None => stream.write_all(b"$-1\r\n").await,
    }
}

async fn redis_array_response(stream: &mut TcpStream, values: &[Vec<u8>]) -> std::io::Result<()> {
    stream
        .write_all(format!("*{}\r\n", values.len()).as_bytes())
        .await?;
    for value in values {
        redis_bulk_response(stream, Some(value)).await?;
    }
    Ok(())
}

fn redis_value_to_string(value: RedisValue) -> Result<String> {
    match value {
        RedisValue::BulkString(bytes) => String::from_utf8(bytes)
            .map_err(|error| WorldForgeError::SerializationError(error.to_string())),
        RedisValue::SimpleString(text) => Ok(text),
        RedisValue::Error(message) => Err(WorldForgeError::InternalError(message)),
        RedisValue::Integer(value) => Ok(value.to_string()),
        RedisValue::Null => Ok(String::new()),
        RedisValue::Array(_) => Err(WorldForgeError::InvalidState(
            "expected Redis string, got nested array".to_string(),
        )),
    }
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}
