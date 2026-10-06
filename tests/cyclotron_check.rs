//! Execute a generated Muon ELF and inspect its exact-output result word.
use cyclotron::base::mem::HasMemory;
use cyclotron::ui::{make_sim, read_toml, CyclotronArgs};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: cyclotron_check CONFIG ELF RESULT_ADDRESS_HEX");
        std::process::exit(2);
    }
    let config_path = PathBuf::from(&args[1]);
    let elf = PathBuf::from(&args[2]);
    let result_addr = usize::from_str_radix(args[3].trim_start_matches("0x"), 16)
        .expect("invalid result address");
    let options = CyclotronArgs {
        config_path: config_path.clone(),
        binary_path: Some(elf),
        timing: false,
        ..CyclotronArgs::default()
    };
    let toml = read_toml(config_path.as_path());
    let mut sim = make_sim(Some(&toml), &Some(options));
    if let Err(code) = sim.simulate() {
        eprintln!("Cyclotron failed or timed out with code {code}");
        std::process::exit(1);
    }
    let memory = sim.top.gmem.read().expect("GPU memory lock poisoned");
    let bytes = memory.read_impl(result_addr, 8).expect("result out of range");
    let core0 = u32::from_le_bytes(bytes[0..4].try_into().expect("core 0 result size"));
    let core1 = u32::from_le_bytes(bytes[4..8].try_into().expect("core 1 result size"));
    println!("MUON_MLIR_RESULT addr=0x{result_addr:08x} core0=0x{core0:08x} core1=0x{core1:08x}");
    if core0 != 0xC0DEFACE || core1 != 0xC0DEFACE {
        std::process::exit(1);
    }
}
