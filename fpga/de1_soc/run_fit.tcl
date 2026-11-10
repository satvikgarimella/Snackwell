# Core-only capacity check. Run with Quartus Prime Lite/Standard:
#   quartus_sh -t fpga/de1_soc/run_fit.tcl
# Virtual bus pins stand in for a future board transport. No bitstream is made.
package require ::quartus::project
package require ::quartus::flow

set script_dir [file dirname [file normalize [info script]]]
set repo_dir [file dirname [file dirname $script_dir]]
set rtl_dir [file join $repo_dir rtl]
set run_dir [file join $repo_dir build quartus-fit \
    "[clock format [clock seconds] -format %Y%m%d-%H%M%S]-[pid]"]
if {![file isfile [file join $rtl_dir snackwell_pkg.sv]]} {
    error "Missing RTL tree: $rtl_dir"
}
if {[file exists $run_dir]} {
    error "Refusing to reuse an existing run directory: $run_dir"
}
file mkdir $run_dir
cd $run_dir

array set status {
    scope core_with_virtual_bus_pins
    target 5CSEMA5F31C6
    analysis_synthesis not_run
    fitter not_run
    timing_report not_run
    timing_verified false
    board_interface_included false
    model_storage_included false
    bitstream_generated false
    hardware_tested false
}
proc save_status {} {
    global status run_dir
    set handle [open [file join $run_dir run-status.txt] w]
    foreach key [lsort [array names status]] {
        puts $handle "$key=$status($key)"
    }
    close $handle
}
save_status

proc add_rtl {directory} {
    foreach filename [lsort [glob -nocomplain -types f -directory $directory *.sv *.v]] {
        if {[file tail $filename] eq "snackwell_pkg.sv"} { continue }
        set kind SYSTEMVERILOG_FILE
        if {[file extension $filename] eq ".v"} { set kind VERILOG_FILE }
        set_global_assignment -name $kind $filename
    }
    foreach child [lsort [glob -nocomplain -types d -directory $directory *]] {
        add_rtl $child
    }
}

project_new snackwell_fit
set_global_assignment -name FAMILY "Cyclone V"
set_global_assignment -name DEVICE 5CSEMA5F31C6
set_global_assignment -name TOP_LEVEL_ENTITY snackwell_device
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files
set_global_assignment -name VERILOG_MACRO "SYNTHESIS=1"
set_global_assignment -name NUM_PARALLEL_PROCESSORS 4
set_global_assignment -name SDC_FILE [file join $script_dir fit_core.sdc]
set_global_assignment -name SYSTEMVERILOG_FILE [file join $rtl_dir snackwell_pkg.sv]
add_rtl $rtl_dir

# Keep clk as a real clock input; the fitter chooses a pin for this capacity
# experiment. All other ports will become internal signals in a board wrapper.
foreach port {
    reset request_valid request_ready request_write
    {request_address[*]} {request_data[*]}
    response_valid response_ready response_error {response_data[*]}
    global_request_valid global_request_ready global_request_write
    {global_request_address[*]} {global_request_data[*]}
    global_response_valid global_response_ready global_response_error {global_response_data[*]}
    busy done fault
} {
    set_instance_assignment -name VIRTUAL_PIN ON -to $port
}
export_assignments

puts "Core fit check for 5CSEMA5F31C6; reports: $run_dir"
foreach {tool field} {map analysis_synthesis fit fitter sta timing_report} {
    if {[catch {execute_module -tool $tool} message]} {
        set status($field) failed
        save_status
        project_close
        puts stderr "$tool failed: $message"
        puts stderr "Inspect $run_dir/output_files and run-status.txt"
        exit 1
    }
    set status($field) completed
    save_status
}
project_close
puts "Core fitting completed. Inspect output_files/snackwell_fit.fit.rpt."
puts "Review timing slack and unconstrained paths in snackwell_fit.sta.rpt."
puts "STA completion alone does not mean timing passed. No board bitstream was made."
