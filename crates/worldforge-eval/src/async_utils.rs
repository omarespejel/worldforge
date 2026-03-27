use std::future::{poll_fn, Future};
use std::pin::Pin;
use std::task::Poll;

pub(crate) type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Drive a batch of futures concurrently while preserving their input order.
pub(crate) async fn join_all_ordered<'a, T>(futures: Vec<BoxFuture<'a, T>>) -> Vec<T> {
    let mut futures: Vec<Option<BoxFuture<'a, T>>> = futures.into_iter().map(Some).collect();
    let mut outputs: Vec<Option<T>> = std::iter::repeat_with(|| None)
        .take(futures.len())
        .collect();

    poll_fn(move |cx| {
        let mut pending = false;

        for index in 0..futures.len() {
            if outputs[index].is_some() {
                continue;
            }

            let Some(future) = futures[index].as_mut() else {
                continue;
            };

            match future.as_mut().poll(cx) {
                Poll::Ready(value) => {
                    outputs[index] = Some(value);
                    futures[index] = None;
                }
                Poll::Pending => pending = true,
            }
        }

        if pending {
            Poll::Pending
        } else {
            Poll::Ready(
                outputs
                    .drain(..)
                    .map(|value| value.expect("join_all_ordered completed without a value"))
                    .collect(),
            )
        }
    })
    .await
}
