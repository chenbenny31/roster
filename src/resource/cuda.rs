//! The few CUDA driver calls qualification needs, loaded at run time from libcuda, so roster needs
//! neither the CUDA toolkit nor a build-time CUDA dependency; the kernels are PTX text that the
//! driver compiles for whatever card it finds

use std::ffi::{c_void, CStr};
use std::ptr::null_mut;

/// An integer hash chain per thread, and a fill and a check pass over a VRAM buffer
const PTX: &CStr = c"
.version 7.0
.target sm_50
.address_size 64

.visible .entry hash(
    .param .u64 hash_out,
    .param .u32 hash_rounds
)
{
    .reg .pred  %p<2>;
    .reg .b32   %r<10>;
    .reg .b64   %rd<5>;

    ld.param.u64        %rd1, [hash_out];
    ld.param.u32        %r1, [hash_rounds];
    mov.u32             %r2, %ctaid.x;
    mov.u32             %r3, %ntid.x;
    mov.u32             %r4, %tid.x;
    mad.lo.u32          %r5, %r2, %r3, %r4;
    mov.u32             %r6, %r5;
    mov.u32             %r7, 0;
hash_loop:
    setp.ge.u32         %p1, %r7, %r1;
    @%p1 bra            hash_done;
    mad.lo.u32          %r6, %r6, 1664525, 1013904223;
    shr.u32             %r8, %r6, 13;
    xor.b32             %r6, %r6, %r8;
    add.u32             %r7, %r7, 1;
    bra                 hash_loop;
hash_done:
    cvta.to.global.u64  %rd2, %rd1;
    mul.wide.u32        %rd3, %r5, 4;
    add.s64             %rd4, %rd2, %rd3;
    st.global.u32       [%rd4], %r6;
    ret;
}

.visible .entry fill(
    .param .u64 fill_buffer,
    .param .u64 fill_words,
    .param .u32 fill_seed
)
{
    .reg .pred  %p<2>;
    .reg .b32   %r<10>;
    .reg .b64   %rd<10>;

    ld.param.u64        %rd1, [fill_buffer];
    ld.param.u64        %rd2, [fill_words];
    ld.param.u32        %r1, [fill_seed];
    cvta.to.global.u64  %rd1, %rd1;
    mov.u32             %r2, %ctaid.x;
    mov.u32             %r3, %ntid.x;
    mov.u32             %r4, %tid.x;
    mad.lo.u32          %r5, %r2, %r3, %r4;
    cvt.u64.u32         %rd3, %r5;
    mov.u32             %r6, %nctaid.x;
    mul.lo.u32          %r6, %r6, %r3;
    cvt.u64.u32         %rd4, %r6;
fill_loop:
    setp.ge.u64         %p1, %rd3, %rd2;
    @%p1 bra            fill_done;
    cvt.u32.u64         %r7, %rd3;
    mul.lo.u32          %r7, %r7, 1664525;
    xor.b32             %r7, %r7, %r1;
    shl.b64             %rd5, %rd3, 2;
    add.s64             %rd6, %rd1, %rd5;
    st.global.u32       [%rd6], %r7;
    add.s64             %rd3, %rd3, %rd4;
    bra                 fill_loop;
fill_done:
    ret;
}

.visible .entry check(
    .param .u64 check_buffer,
    .param .u64 check_words,
    .param .u32 check_seed,
    .param .u64 check_errors
)
{
    .reg .pred  %p<3>;
    .reg .b32   %r<10>;
    .reg .b64   %rd<10>;

    ld.param.u64        %rd1, [check_buffer];
    ld.param.u64        %rd2, [check_words];
    ld.param.u32        %r1, [check_seed];
    ld.param.u64        %rd7, [check_errors];
    cvta.to.global.u64  %rd1, %rd1;
    cvta.to.global.u64  %rd7, %rd7;
    mov.u32             %r2, %ctaid.x;
    mov.u32             %r3, %ntid.x;
    mov.u32             %r4, %tid.x;
    mad.lo.u32          %r5, %r2, %r3, %r4;
    cvt.u64.u32         %rd3, %r5;
    mov.u32             %r6, %nctaid.x;
    mul.lo.u32          %r6, %r6, %r3;
    cvt.u64.u32         %rd4, %r6;
check_loop:
    setp.ge.u64         %p1, %rd3, %rd2;
    @%p1 bra            check_done;
    cvt.u32.u64         %r7, %rd3;
    mul.lo.u32          %r7, %r7, 1664525;
    xor.b32             %r7, %r7, %r1;
    shl.b64             %rd5, %rd3, 2;
    add.s64             %rd6, %rd1, %rd5;
    ld.global.u32       %r8, [%rd6];
    setp.eq.u32         %p2, %r8, %r7;
    @%p2 bra            check_next;
    atom.global.add.u32 %r9, [%rd7], 1;
check_next:
    add.s64             %rd3, %rd3, %rd4;
    bra                 check_loop;
check_done:
    ret;
}
";

const HASH_ROUNDS: u32 = 1000;
const THREADS: u32 = 256;            // per block; blocks = 4 per SM
const MAX_TEST_BYTES: usize = 1 << 30; // VRAM pattern test covers up to 1 GB, at most half of what's free
const MULTIPROCESSOR_COUNT: i32 = 16; // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT

type Status = i32;

/// The driver entry points, looked up in libcuda.so.1
struct Driver {
    init:         unsafe extern "C" fn(u32) -> Status,
    device_get:   unsafe extern "C" fn(*mut i32, i32) -> Status,
    attribute:    unsafe extern "C" fn(*mut i32, i32, i32) -> Status,
    ctx_create:   unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> Status,
    ctx_destroy:  unsafe extern "C" fn(*mut c_void) -> Status,
    module_load:  unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> Status,
    function:     unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const libc::c_char) -> Status,
    mem_info:     unsafe extern "C" fn(*mut usize, *mut usize) -> Status,
    alloc:        unsafe extern "C" fn(*mut u64, usize) -> Status,
    memset:       unsafe extern "C" fn(u64, u32, usize) -> Status,
    copy_to_host: unsafe extern "C" fn(*mut c_void, u64, usize) -> Status,
    launch:       unsafe extern "C" fn(*mut c_void, u32, u32, u32, u32, u32, u32, u32, *mut c_void,
                                       *mut *mut c_void, *mut *mut c_void) -> Status,
    synchronize:  unsafe extern "C" fn() -> Status,
}

impl Driver {
    /// None when this machine has no NVIDIA driver
    fn load() -> Option<Driver> {
        unsafe {
            let library = libc::dlopen(c"libcuda.so.1".as_ptr(), libc::RTLD_NOW);
            if library.is_null() {
                return None;
            }
            macro_rules! symbol {
                ($name:literal) => {{
                    let pointer = libc::dlsym(library, concat!($name, "\0").as_ptr() as *const libc::c_char);
                    if pointer.is_null() {
                        return None;
                    }
                    std::mem::transmute(pointer)
                }};
            }
            Some(Driver {
                init:         symbol!("cuInit"),
                device_get:   symbol!("cuDeviceGet"),
                attribute:    symbol!("cuDeviceGetAttribute"),
                ctx_create:   symbol!("cuCtxCreate_v2"),
                ctx_destroy:  symbol!("cuCtxDestroy_v2"),
                module_load:  symbol!("cuModuleLoadData"),
                function:     symbol!("cuModuleGetFunction"),
                mem_info:     symbol!("cuMemGetInfo_v2"),
                alloc:        symbol!("cuMemAlloc_v2"),
                memset:       symbol!("cuMemsetD32_v2"),
                copy_to_host: symbol!("cuMemcpyDtoH_v2"),
                launch:       symbol!("cuLaunchKernel"),
                synchronize:  symbol!("cuCtxSynchronize"),
            })
        }
    }
}

fn check(status: Status, call: &str) -> Result<(), String> {
    if status == 0 { Ok(()) } else { Err(format!("{} failed with CUDA error {}", call, status)) }
}

/// Run the kernels on CUDA device 0, which CUDA_VISIBLE_DEVICES makes the card under test;
/// None when there is no driver to run them with, e.g. on a simulated node
pub fn kernel_test() -> Option<Result<(), String>> {
    let cuda = Driver::load()?;
    Some(unsafe { with_context(&cuda) })
}

/// Destroying the context frees everything the test allocated, on any path out
unsafe fn with_context(cuda: &Driver) -> Result<(), String> {
    check((cuda.init)(0), "cuInit")?;
    let mut device = 0;
    check((cuda.device_get)(&mut device, 0), "cuDeviceGet")?;
    let mut sms = 0;
    check((cuda.attribute)(&mut sms, MULTIPROCESSOR_COUNT, device), "cuDeviceGetAttribute")?;
    let mut context = null_mut();
    check((cuda.ctx_create)(&mut context, 0, device), "cuCtxCreate")?;
    let result = test(cuda, sms as u32);
    (cuda.ctx_destroy)(context);
    result
}

unsafe fn test(cuda: &Driver, sms: u32) -> Result<(), String> {
    let mut module = null_mut();
    check((cuda.module_load)(&mut module, PTX.as_ptr() as *const c_void), "cuModuleLoadData")?;
    let kernel = |name: &CStr| -> Result<*mut c_void, String> {
        let mut function = null_mut();
        check((cuda.function)(&mut function, module, name.as_ptr()), "cuModuleGetFunction")?;
        Ok(function)
    };
    let blocks = sms * 4;
    let run = |function: *mut c_void, params: &mut [*mut c_void], name: &str| -> Result<(), String> {
        check((cuda.launch)(function, blocks, 1, 1, THREADS, 1, 1, 0, null_mut(), params.as_mut_ptr(), null_mut()), name)?;
        check((cuda.synchronize)(), name)
    };

    // every SM runs hash chains; the CPU computes the same chains and every result must match
    let count = (blocks * THREADS) as usize;
    let mut out = 0u64;
    check((cuda.alloc)(&mut out, count * 4), "cuMemAlloc")?;
    let mut rounds = HASH_ROUNDS;
    run(kernel(c"hash")?, &mut [&mut out as *mut u64 as *mut c_void, &mut rounds as *mut u32 as *mut c_void], "hash kernel")?;
    let mut results = vec![0u32; count];
    check((cuda.copy_to_host)(results.as_mut_ptr() as *mut c_void, out, count * 4), "cuMemcpyDtoH")?;
    let wrong = results.iter().enumerate().filter(|(index, value)| **value != hash(*index as u32, HASH_ROUNDS)).count();
    if wrong > 0 {
        return Err(format!("{} of {} hash chains wrong", wrong, count));
    }

    // a pattern and its inverse over up to 1 GB of VRAM, so a stuck bit shows either way
    let (mut free, mut total) = (0usize, 0usize);
    check((cuda.mem_info)(&mut free, &mut total), "cuMemGetInfo")?;
    let bytes = (free / 2).min(MAX_TEST_BYTES) & !3;
    let (mut buffer, mut errors) = (0u64, 0u64);
    check((cuda.alloc)(&mut buffer, bytes), "cuMemAlloc")?;
    check((cuda.alloc)(&mut errors, 4), "cuMemAlloc")?;
    check((cuda.memset)(errors, 0, 1), "cuMemsetD32")?;
    let mut words = (bytes / 4) as u64;
    for mut seed in [0u32, u32::MAX] {
        let buffer_param = &mut buffer as *mut u64 as *mut c_void;
        let words_param = &mut words as *mut u64 as *mut c_void;
        let seed_param = &mut seed as *mut u32 as *mut c_void;
        run(kernel(c"fill")?, &mut [buffer_param, words_param, seed_param], "fill kernel")?;
        run(kernel(c"check")?, &mut [buffer_param, words_param, seed_param, &mut errors as *mut u64 as *mut c_void], "check kernel")?;
    }
    let mut mismatches = 0u32;
    check((cuda.copy_to_host)(&mut mismatches as *mut u32 as *mut c_void, errors, 4), "cuMemcpyDtoH")?;
    if mismatches > 0 {
        return Err(format!("{} of {} VRAM words read back wrong", mismatches, words * 2));
    }
    Ok(())
}

/// The CPU side of the hash kernel
fn hash(index: u32, rounds: u32) -> u32 {
    let mut value = index;
    for _ in 0..rounds {
        value = value.wrapping_mul(1664525).wrapping_add(1013904223);
        value ^= value >> 13;
    }
    value
}

// tests

#[cfg(test)]
mod tests {
    use super::*;

    // uses the real GPU when there is one: cargo test -- --ignored kernel_test_passes
    #[test]
    #[ignore]
    fn kernel_test_passes_on_this_machine() {
        if let Some(result) = kernel_test() {
            assert_eq!(result, Ok(()));
        }
    }

    #[test]
    fn hash_matches_the_kernel_arithmetic() {
        // one round of x * 1664525 + 1013904223, then x ^= x >> 13
        let x = 1013904223u32;
        assert_eq!(hash(0, 1), x ^ (x >> 13));
    }
}
