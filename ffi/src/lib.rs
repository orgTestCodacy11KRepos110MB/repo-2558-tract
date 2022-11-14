use anyhow::Context;
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::sync::Arc;

use tract_nnef::internal as native;
use tract_nnef::tract_core::prelude::*;

/// Used as a return type of functions that can encounter errors.
/// If the function encountered an error, you can retrieve it using the `tract_get_last_error`
/// function
#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Debug, PartialEq, Eq)]
pub enum TRACT_RESULT {
    /// The function returned successfully
    TRACT_RESULT_OK = 0,
    /// The function returned an error
    TRACT_RESULT_KO = 1,
}

thread_local! {
    pub(crate) static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
}

fn wrap<F: FnOnce() -> anyhow::Result<()>>(func: F) -> TRACT_RESULT {
    match func() {
        Ok(_) => TRACT_RESULT::TRACT_RESULT_OK,
        Err(e) => {
            let msg = format!("{:?}", e);
            if std::env::var("TRACT_ERROR_STDERR").is_ok() {
                eprintln!("{}", msg);
            }
            LAST_ERROR.with(|p| {
                *p.borrow_mut() = Some(CString::new(msg).unwrap_or_else(|_| {
                    CString::new("tract error message contains 0, can't convert to CString")
                        .unwrap()
                }))
            });
            TRACT_RESULT::TRACT_RESULT_KO
        }
    }
}

/// Used to retrieve the last error that happened in this thread. A function encountered an error if
/// its return type is of type `TRACT_RESULT` and it returned `TRACT_RESULT_KO`.
///
/// # Return value
///  It returns a pointer to a null-terminated UTF-8 string that will contain the error description.
///  Rust side keeps ownership of the buffer. It will be valid as long as no other tract calls is
///  performed by the thread.
///  If no error occured, null is returned.
#[no_mangle]
pub extern "C" fn tract_get_last_error() -> *const std::ffi::c_char {
    LAST_ERROR.with(|msg| msg.borrow().as_ref().map(|s| s.as_ptr()).unwrap_or(std::ptr::null()))
}

// NNEF

pub struct TractNnef(native::Nnef);

#[no_mangle]
pub extern "C" fn tract_nnef_create(nnef: *mut *mut TractNnef) -> TRACT_RESULT {
    wrap(|| unsafe {
        *nnef = Box::into_raw(Box::new(TractNnef(tract_nnef::nnef())));
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn tract_nnef_destroy(nnef: *mut *mut TractNnef) -> TRACT_RESULT {
    wrap(|| unsafe {
        if nnef.is_null() || (*nnef).is_null() {
            anyhow::bail!("Trying to destroy a null Nnef object");
        }
        let _ = Box::from_raw(*nnef);
        *nnef = std::ptr::null_mut();
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn tract_nnef_model_for_path(
    nnef: &TractNnef,
    path: *const c_char,
    model: *mut *mut TractModel,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        *model = std::ptr::null_mut();
        let path = CStr::from_ptr(path).to_str()?;
        let m = Box::new(TractModel(
            nnef.0.model_for_path(path).with_context(|| format!("opening file {:?}", path))?,
        ));
        *model = Box::into_raw(m);
        Ok(())
    })
}

// TYPED MODEL

pub struct TractModel(TypedModel);

#[no_mangle]
pub extern "C" fn tract_model_optimize(model: *mut TractModel) -> TRACT_RESULT {
    wrap(|| unsafe {
        if let Some(model) = model.as_mut() {
            model.0.optimize()
        } else {
            anyhow::bail!("Trying to optimise null model")
        }
    })
}

/// Convert a TypedModel into a TypedRunnableModel.
///
/// This function transfers ownership of the model argument to the runnable model.
#[no_mangle]
pub extern "C" fn tract_model_into_runnable(
    model: *mut *mut TractModel,
    runnable: *mut *mut TractRunnable,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        if model.is_null() {
            anyhow::bail!("Trying to convert null model")
        }
        let m = Box::from_raw(*model);
        *model = std::ptr::null_mut();
        let runnable_model = m.0.into_runnable()?;
        *runnable = Box::into_raw(Box::new(TractRunnable(Arc::new(runnable_model)))) as _;
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn tract_model_destroy(model: *mut *mut TractModel) -> TRACT_RESULT {
    wrap(|| unsafe {
        if model.is_null() || (*model).is_null() {
            anyhow::bail!("Trying to destroy a null Model");
        }
        let _ = Box::from_raw(*model);
        Ok(())
    })
}

// RUNNABLE MODEL
pub struct TractRunnable(Arc<native::TypedRunnableModel<native::TypedModel>>);

#[no_mangle]
pub extern "C" fn tract_runnable_spawn_state(
    runnable: *mut TractRunnable,
    state: *mut *mut TractState,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        if state.is_null() {
            anyhow::bail!("Null pointer for expected state return")
        }
        *state = std::ptr::null_mut();
        if runnable.is_null() {
            anyhow::bail!("Trying to convert null model")
        }
        let s = native::TypedSimpleState::new((*runnable).0.clone())?;
        *state = Box::into_raw(Box::new(TractState(s)));
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn tract_runnable_release(model: *mut *mut TractRunnable) -> TRACT_RESULT {
    wrap(|| unsafe {
        if model.is_null() || (*model).is_null() {
            anyhow::bail!("Trying to destroy a null Runnable");
        }
        let _ = Box::from_raw(model);
        Ok(())
    })
}

/*
#[no_mangle]
pub extern "C" fn tract_foo(model: *mut TractRunnable) -> TRACT_RESULT {
    unsafe {
    }
    TRACT_RESULT::TRACT_RESULT_OK
}
*/

// STATE
/// cbindgen:ignore
type NativeState = native::TypedSimpleState<
    native::TypedModel,
    Arc<native::TypedRunnableModel<native::TypedModel>>,
>;
pub struct TractState(NativeState);
