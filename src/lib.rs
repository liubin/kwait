use std::time::Duration;

#[macro_use(defer)]
extern crate scopeguard;

use crossbeam::channel::Receiver;
use crossbeam::crossbeam_channel::{select, tick, unbounded};
use crossbeam::sync::WaitGroup;
use std::fmt;

use std::error::Error;
use std::result;
use std::thread;

use rand::Rng;
use std::time::Instant;

type Result = result::Result<(), Box<dyn Error>>;

// Group allows to start a group of goroutines and wait for their completion.
pub struct Group {
    wg: WaitGroup,
}

impl Group {
    pub fn new() -> Group {
        Group {
            wg: WaitGroup::new(),
        }
    }

    pub fn wait(&self) {
        self.wg.clone().wait();
    }

    // start_with_channel starts f in a new goroutine in the group.
    // stop_ch is passed to f as an argument. f should stop when stop_ch is available.
    pub fn start_with_channel<F>(&self, stop_ch: Receiver<bool>, f: F)
    where
        F: Fn(Receiver<bool>) -> () + 'static + std::marker::Send + std::marker::Sync,
    {
        self.start(move || f(stop_ch.clone()));
    }

    // start starts f in a new goroutine in the group.
    pub fn start<F>(&self, f: F)
    where
        F: Fn() -> () + std::marker::Send + 'static,
    {
        let wg = self.wg.clone();
        thread::spawn(move || {
            f();
            drop(wg);
        });
    }
}

// forever calls f every period for ever.
//
// forever is syntactic sugar on top of until.
pub fn forever<F>(f: F, period: Duration)
where
    F: Fn() -> (),
{
    let (_s, r) = unbounded();
    until(f, period, r)
}

// until loops until stop channel is closed, running f every period.
//
// until is syntactic sugar on top of jitter_until with zero jitter factor and
// with sliding = true (which means the timer for period starts after the f
// completes).
pub fn until<F>(f: F, period: Duration, stop_ch: Receiver<bool>)
where
    F: Fn() -> (),
{
    jitter_until(f, period, 0.0, true, stop_ch)
}

// non_sliding_until loops until stop channel is closed, running f every
// period.
//
// non_sliding_until is syntactic sugar on top of jitter_until with zero jitter
// factor, with sliding = false (meaning the timer for period starts at the same
// time as the function starts).
pub fn non_sliding_until<F>(f: F, period: Duration, stop_ch: Receiver<bool>)
where
    F: Fn() -> (),
{
    jitter_until(f, period, 0.0, false, stop_ch)
}

// jitter_until loops until stop channel is closed, running f every period.
//
// If jitter_factor is positive, the period is jittered before every run of f.
// If jitter_factor is not positive, the period is unchanged and not jittered.
//
// If sliding is true, the period is computed after f runs. If it is false then
// period includes the runtime for f.
//
// Close stop_ch to stop. f may not be invoked if stop channel is already
// closed. Pass NeverStop to if you don't want it stop.
pub fn jitter_until<F>(
    f: F,
    period: Duration,
    jitter_factor: f64,
    sliding: bool,
    stop_ch: Receiver<bool>,
) where
    F: Fn() -> (),
{
    backoff_until(
        f,
        JitteredBackoffManagerImpl::new_jittered_backoff_manager(period, jitter_factor),
        sliding,
        stop_ch,
    )
}

// backoff_until loops until stop channel is closed, run f every duration given by BackoffManager.
//
// If sliding is true, the period is computed after f runs. If it is false then
// period includes the runtime for f.
pub fn backoff_until<F>(
    f: F,
    mut backoff: Box<dyn BackoffManager>,
    sliding: bool,
    stop_ch: Receiver<bool>,
) where
    F: Fn() -> (),
{
    loop {
        select! {
            recv(stop_ch) -> _ => return ,
            default => {}
        }

        let mut t = backoff.backoff();

        // FIXME handle crach
        // func() {
        //     defer runtime.HandleCrash()
        //     f()
        // }()
        f();
        if sliding {
            t = backoff.backoff();
        }

        // NOTE: b/c there is no priority selection in golang
        // it is possible for this to race, meaning we could
        // trigger t.C and stop_ch, and t.C select falls through.
        // In order to mitigate we re-check stop_ch at the beginning
        // of every loop to prevent extra executions of f().
        select! {
            recv(stop_ch) -> _ =>  return,
            recv(t) ->  _msg => {            }
        }
    }
}

// jitter returns a Duration between duration and duration + max_factor *
// duration.
//
// This allows clients to avoid converging on periodic behavior. If max_factor
// is 0.0, a suggested default value will be chosen.
pub fn jitter(duration: Duration, max_factor: f64) -> Duration {
    let mut mf = max_factor;
    if mf <= 0.0 {
        mf = 1.0;
    }
    let mut rng = rand::thread_rng();
    Duration::from_nanos((duration.as_nanos() as f64 * (1.0 + rng.gen::<u64>() as f64 * mf)) as u64)
}

// WaitTimeoutError is returned when the condition exited without success.
#[derive(Debug, Clone)]
struct WaitTimeoutError;

impl fmt::Display for WaitTimeoutError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "timed out waiting for the condition")
    }
}
impl Error for WaitTimeoutError {}

// run_condition_with_crash_protection runs a ConditionFunc with crash protection
fn run_condition_with_crash_protection<F>(condition: F) -> std::result::Result<bool, Box<dyn Error>>
where
    F: Fn() -> std::result::Result<bool, Box<dyn Error>> + Copy,
{
    // defer runtime.HandleCrash()
    condition()
}

// Backoff holds parameters applied to a Backoff function.
pub struct Backoff {
    // The initial duration.
    duration: Duration,
    // Duration is multiplied by factor each iteration, if factor is not zero
    // and the limits imposed by steps and Cap have not been reached.
    // Should not be negative.
    // The jitter does not contribute to the updates to the duration parameter.
    factor: f64,
    // The sleep at each iteration is the duration plus an additional
    // amount chosen uniformly at random from the interval between
    // zero and `jitter*duration`.
    jitter: f64,
    // The remaining number of iterations in which the duration
    // parameter may change (but progress can be stopped earlier by
    // hitting the cap). If not positive, the duration is not
    // changed. Used for exponential backoff in combination with
    // Factor and Cap.
    steps: i32,
    // A limit on revised values of the duration parameter. If a
    // multiplication by the factor parameter would make the duration
    // exceed the cap then the duration is set to the cap and the
    // steps parameter is set to zero.
    cap: Duration,
}

impl Backoff {
    // step (1) returns an amount of time to sleep determined by the
    // original Duration and jitter and (2) mutates the provided Backoff
    // to update its steps and Duration.
    pub fn step(&mut self) -> Duration {
        if self.steps < 1 {
            if self.jitter > 0.0 {
                return jitter(self.duration, self.jitter);
            }
            return self.duration;
        }
        self.steps = self.steps - 1;

        let mut duration = self.duration;

        // calculate the next step
        if self.factor != 0.0 {
            self.duration =
                Duration::from_nanos((self.duration.as_nanos() as f64 * self.factor) as u64);
            if !(self.cap.as_nanos() == 0) && self.duration > self.cap {
                self.duration = self.cap;
                self.steps = 0;
            }
        }

        if self.jitter > 0.0 {
            duration = jitter(duration, self.jitter);
        }

        duration
    }
}

// BackoffManager manages backoff with a particular scheme based on its underlying implementation. It provides
// an interface to return a timer for backoff, and caller shall backoff until Timer.C() drains. If the second backoff()
// is called before the timer from the first backoff() call finishes, the first timer will NOT be drained and result in
// undetermined behavior.
// The BackoffManager is supposed to be called in a single-threaded environment.
pub trait BackoffManager {
    fn backoff(&mut self) -> Receiver<Instant>;
}

struct ExponentialBackoffManagerImpl {
    backoff: Backoff,
    last_backoff_start: Instant,
    initial_backoff: Duration,
    backoff_reset_duration: Duration,
}

impl BackoffManager for ExponentialBackoffManagerImpl {
    // Backoff implements BackoffManager.Backoff, it returns a timer so caller can block on the timer for exponential backoff.
    // The returned timer must be drained before calling backoff() the second time
    fn backoff(&mut self) -> Receiver<Instant> {
        tick(self.get_next_backoff())
    }
}

impl ExponentialBackoffManagerImpl {
    // new_exponential_backoff_manager returns a manager for managing exponential backoff. Each backoff is jittered and
    // backoff will not exceed the given max. If the backoff is not called within resetDuration, the backoff is reset.
    // This backoff manager is used to reduce load during upstream unhealthiness.
    pub fn new_exponential_backoff_manager(
        init_backoff: Duration,
        max_backoff: Duration,
        reset_duration: Duration,
        backoff_factor: f64,
        jitter: f64,
    ) -> Box<dyn BackoffManager> {
        Box::new(ExponentialBackoffManagerImpl {
            backoff: Backoff {
                duration: init_backoff,
                factor: backoff_factor,
                jitter: jitter,

                // the current impl of wait.Backoff returns Backoff.Duration once steps are used up, which is not
                // what we ideally need here, we set it to max int and assume we will never use up the steps
                steps: std::i32::MAX,
                cap: max_backoff,
            },
            initial_backoff: init_backoff,
            last_backoff_start: Instant::now(),
            backoff_reset_duration: reset_duration,
        })
    }

    fn get_next_backoff(&mut self) -> Duration {
        if Instant::now().duration_since(self.last_backoff_start) > self.backoff_reset_duration {
            self.backoff.steps = std::i32::MAX;
            self.backoff.duration = self.initial_backoff;
        }
        self.last_backoff_start = Instant::now();
        return self.backoff.step();
    }
}

struct JitteredBackoffManagerImpl {
    duration: Duration,
    jitter: f64,
}

impl BackoffManager for JitteredBackoffManagerImpl {
    // Backoff implements BackoffManager.Backoff, it returns a timer so caller can block on the timer for jittered backoff.
    // The returned timer must be drained before calling backoff() the second time
    fn backoff(&mut self) -> Receiver<Instant> {
        tick(self.get_next_backoff())
    }
}

impl JitteredBackoffManagerImpl {
    // new_jittered_backoff_manager returns a BackoffManager that backoffs with given duration plus given jitter.
    // If the jitter
    // is negative, backoff will not be jittered.
    pub fn new_jittered_backoff_manager(
        duration: Duration,
        jitter: f64,
    ) -> Box<dyn BackoffManager> {
        Box::new(JitteredBackoffManagerImpl {
            duration: duration,
            jitter: jitter,
        }) as Box<dyn BackoffManager>
    }

    fn get_next_backoff(&self) -> Duration {
        if self.jitter > 0.0 {
            jitter(self.duration, self.jitter)
        } else {
            self.duration
        }
    }
}

// exponential_backoff repeats a condition check with exponential backoff.
//
// It repeatedly checks the condition and then sleeps, using `backoff.step()`
// to determine the length of the sleep and adjust Duration and steps.
// Stops and returns as soon as:
// 1. the condition check returns true or an error,
// 2. `backoff.steps` checks of the condition have been done, or
// 3. a sleep truncated by the cap on duration has been completed.
// In case (1) the returned error is what the condition function returned.
// In all other cases, WaitTimeoutError is returned.
pub fn exponential_backoff<F>(backoff: &mut Backoff, condition: F) -> Result
where
    F: Fn() -> std::result::Result<bool, Box<dyn Error>> + Copy,
{
    while backoff.steps > 0 {
        let ok = run_condition_with_crash_protection(condition)?;
        if ok {
            return Ok(());
        }
        if backoff.steps == 1 {
            break;
        }
        thread::sleep(backoff.step());
    }
    Err(Box::new(WaitTimeoutError))
}

// poll tries a condition func until it returns true, an error, or the timeout
// is reached.
//
// poll always waits the interval before the run of 'condition'.
// 'condition' will always be invoked at least once.
//
// Some intervals may be missed if the condition takes too long or the time
// window is too short.
//
// If you want to poll something forever, see poll_infinite.
pub fn poll<F>(interval: Duration, timeout: Duration, condition: F) -> Result
where
    F: Fn() -> std::result::Result<bool, Box<dyn Error>> + Copy,
{
    poll_internal(poller(interval, timeout), condition)
}

fn poll_internal<F>(wait: Box<dyn Fn(Receiver<bool>) -> Receiver<bool>>, condition: F) -> Result
where
    F: Fn() -> std::result::Result<bool, Box<dyn Error>> + Copy,
{
    let (_s, r) = unbounded();
    wait_for(wait, condition, r)
}

// poll_immediate tries a condition func until it returns true, an error, or the timeout
// is reached.
//
// poll_immediate always checks 'condition' before waiting for the interval. 'condition'
// will always be invoked at least once.
//
// Some intervals may be missed if the condition takes too long or the time
// window is too short.
//
// If you want to immediately poll something forever, see poll_immediate_infinite.
pub fn poll_immediate<F>(interval: Duration, timeout: Duration, condition: F) -> Result
where
    F: Fn() -> std::result::Result<bool, Box<dyn Error>> + Copy,
{
    poll_immediate_internal(poller(interval, timeout), condition)
}

fn poll_immediate_internal<F>(
    wait: Box<dyn Fn(Receiver<bool>) -> Receiver<bool>>,
    condition: F,
) -> Result
where
    F: Fn() -> std::result::Result<bool, Box<dyn Error>> + Copy,
{
    let done = run_condition_with_crash_protection(condition)?;
    if done {
        return Ok(());
    }
    poll_internal(wait, condition)
}

// poll_infinite tries a condition func until it returns true or an error
//
// poll_infinite always waits the interval before the run of 'condition'.
//
// Some intervals may be missed if the condition takes too long or the time
// window is too short.
pub fn poll_infinite<F>(interval: Duration, condition: F) -> Result
where
    F: Fn() -> std::result::Result<bool, Box<dyn Error>> + Copy,
{
    let (_s, r) = unbounded();
    return poll_until(interval, condition, r);
}

// poll_immediate_infinite tries a condition func until it returns true or an error
//
// poll_immediate_infinite runs the 'condition' before waiting for the interval.
//
// Some intervals may be missed if the condition takes too long or the time
// window is too short.
pub fn poll_immediate_infinite<F>(interval: Duration, condition: F) -> Result
where
    F: Fn() -> std::result::Result<bool, Box<dyn Error>> + Copy,
{
    let done = run_condition_with_crash_protection(condition)?;
    if done {
        return Ok(());
    }
    poll_infinite(interval, condition)
}

// poll_until tries a condition func until it returns true, an error or stop_ch is
// closed.
//
// poll_until always waits interval before the first run of 'condition'.
// 'condition' will always be invoked at least once.
pub fn poll_until<F>(interval: Duration, condition: F, stop_ch: Receiver<bool>) -> Result
where
    F: Fn() -> std::result::Result<bool, Box<dyn Error>> + Copy,
{
    return wait_for(poller(interval, Duration::new(0, 0)), condition, stop_ch);
}

// poll_immediate_until tries a condition func until it returns true, an error or stop_ch is closed.
//
// poll_immediate_until runs the 'condition' before waiting for the interval.
// 'condition' will always be invoked at least once.
pub fn poll_immediate_until<F>(interval: Duration, condition: F, stop_ch: Receiver<bool>) -> Result
where
    F: Fn() -> std::result::Result<bool, Box<dyn Error>> + Copy,
{
    let done = condition()?;
    if done {
        return Ok(());
    }

    select! {
        recv(stop_ch) -> _ =>   return Err(Box::new(WaitTimeoutError)) ,
        default => return poll_until(interval, condition, stop_ch)
    }
}

// wait_for continually checks 'fn' as driven by 'wait'.
//
// wait_for gets a channel from 'wait()', and then invokes 'fn' once for every value
// placed on the channel and once more when the channel is closed. If the channel is closed
// and 'fn' returns false without error, wait_for returns WaitTimeoutError.
//
// If 'fn' returns an error the loop ends and that error is returned. If
// 'fn' returns true the loop ends and nil is returned.
//
// WaitTimeoutError will be returned if the 'done' channel is closed without fn ever
// returning true.
//
// When the done channel is closed, because the golang `select` statement is
// "uniform pseudo-random", the `fn` might still run one or multiple time,
// though eventually `wait_for` will return.
pub fn wait_for<F>(
    wait: Box<dyn Fn(Receiver<bool>) -> Receiver<bool>>,
    func: F,
    done: Receiver<bool>,
) -> Result
where
    F: Fn() -> std::result::Result<bool, Box<dyn Error>> + Copy,
{
    let (s, r) = unbounded();
    let c = wait(r);
    // notify wait thread to quit.
    defer! { drop(s);};
    loop {
        select! {
           recv(c) -> msg => {
            let ok = run_condition_with_crash_protection(func)?;
            if ok {
                return Ok(());
            }
            if msg.is_err() {
                return Err(Box::new(WaitTimeoutError));
            }
        },
            recv(done) -> _ => return Err(Box::new(WaitTimeoutError)),
        }
    }
}

// poller returns a Box<dyn Fn(Receiver<bool>) -> Receiver<bool>>
// that will send to the channel every interval until
// timeout has elapsed and then closes the channel.
//
// Over very short intervals you may receive no ticks before the channel is
// closed. A timeout of 0.0 is interpreted as an infinity, and in such a case
// it would be the caller's responsibility to close the done channel.
// Failure to do so would result in a leaked goroutine.
//
// Output ticks are not buffered. If the channel is not ready to receive an
// item, the tick is skipped.
fn poller(interval: Duration, timeout: Duration) -> Box<dyn Fn(Receiver<bool>) -> Receiver<bool>> {
    let func = move |done: Receiver<bool>| -> Receiver<bool> {
        let (s, r) = unbounded();
        let rr = r.clone();
        thread::spawn(move || {
            let ticker = tick(interval);

            // FIXME: workaround for compile error use of possibly-uninitialized `after`
            let mut after = tick(Duration::from_secs(1000000000));
            if !(timeout.as_nanos() == 0) {
                after = tick(timeout);
            }

            loop {
                select! {
                    recv(ticker) -> _ => {
                        // If the consumer isn't ready for this signal drop it and
                        // check the other channels.
                        s.send(true).unwrap();
                    },
                    recv(after) -> _ => {
                        return
                    },
                    recv(done) -> _ => {
                        return
                    },
                }
            }
        });

        rr
    };

    Box::new(func)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_poll() {
        let (s, r) = unbounded();
        let cond_fn = || {
            select! {
                recv(r) -> msg => {
                    if msg.unwrap() == 3 {
                        return Ok(true);
                    }else {
                        return Ok(false);
                    }
                },
                default => {
                    return Ok(false)
                }
                    ,
            }
        };

        thread::spawn(move || {
            for i in 1..4 {
                thread::sleep(Duration::from_millis(20));
                s.send(i).unwrap();
            }
            drop(s);
            println!("sender dropped");
        });

        let ret = poll(
            Duration::from_millis(100),
            Duration::from_millis(300),
            cond_fn,
        );
        assert_eq!(true, ret.is_ok());

        let cond_fn = || Ok(false);
        let ret = poll(
            Duration::from_millis(100),
            Duration::from_millis(300),
            cond_fn,
        );

        assert_eq!(true, ret.is_err());
    }

    #[test]
    fn test_until() {
        let (s, r) = unbounded();

        let counter = Arc::new(Mutex::new(0));

        let counter1 = counter.clone();
        let worker_fn = move || {
            let mut counter = counter1.lock().unwrap();
            *counter += 1;
            println!("do work {}", *counter);
        };

        let counter2 = counter.clone();
        std::thread::spawn(move || loop {
            {
                let counter = counter2.lock().unwrap();
                if *counter > 4 {
                    drop(s);
                    println!("sender dropped");
                    return;
                }
            }

            thread::sleep(Duration::from_millis(10));
        });

        until(worker_fn, Duration::from_millis(10), r);
        let counter = counter.lock().unwrap();
        println!("final counter {}", *counter);
        assert_eq!(true, *counter > 4);
    }

    #[test]
    fn test_xxx() {
        let zero_seconds = Duration::new(0, 0);
        assert_eq!(0, zero_seconds.as_nanos());

        let (s, r) = unbounded();

        thread::spawn(move || {
            s.send(1).unwrap();
            s.send(2).unwrap();
        });

        let msg1 = r.recv().unwrap();
        let msg2 = r.recv().unwrap();

        assert_eq!(msg1 + msg2, 3);
    }
}
