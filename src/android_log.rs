#[cfg(target_os = "android")]
use std::ffi::{c_char, CString};

const ANDROID_LOG_INFO: i32 = 4;
const ANDROID_LOG_ERROR: i32 = 6;
const LOG_TAG: &str = "SayboardParakeetRs";

#[cfg(target_os = "android")]
unsafe extern "C" {
    fn __android_log_write(prio: i32, tag: *const c_char, text: *const c_char) -> i32;
}

fn log_with_priority(priority: i32, message: &str) {
    let safe_message = message.replace('\0', " ");
    #[cfg(target_os = "android")]
    {
        let tag = CString::new(LOG_TAG).expect("valid android log tag");
        if let Ok(text) = CString::new(safe_message) {
            unsafe {
                __android_log_write(priority, tag.as_ptr(), text.as_ptr());
            }
        }
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = priority;
        eprintln!("{LOG_TAG}: {safe_message}");
    }
}

pub(crate) fn info(message: impl AsRef<str>) {
    log_with_priority(ANDROID_LOG_INFO, message.as_ref());
}

pub(crate) fn error(message: impl AsRef<str>) {
    log_with_priority(ANDROID_LOG_ERROR, message.as_ref());
}
