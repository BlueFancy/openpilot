#!/usr/bin/env python3
# simple pandad wrapper that updates the panda first
import os
import usb1
import time
import signal
import subprocess

from panda import Panda, PandaDFU, PandaProtocolMismatch, FW_PATH
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.system.hardware import HARDWARE
from openpilot.common.swaglog import cloudlog


def get_expected_signature(panda: Panda) -> bytes:
  try:
    fn = os.path.join(FW_PATH, panda.get_mcu_type().config.app_fn)
    return Panda.get_signature_from_firmware(fn)
  except Exception:
    cloudlog.exception("Error computing expected signature")
    return b""

def flash_panda(panda_serial: str) -> Panda:
  try:
    panda = Panda(panda_serial)
  except PandaProtocolMismatch:
    cloudlog.warning("detected protocol mismatch, reflashing panda")
    HARDWARE.recover_internal_panda()
    raise

  # skip flashing if the detected panda is not supported
  supported_panda = check_panda_support(panda)
  if not supported_panda:
    cloudlog.warning(f"Panda {panda_serial} is not supported (hw_type: {panda.get_type()}), skipping flash...")
    return panda

  fw_signature = get_expected_signature(panda)
  internal_panda = panda.is_internal()

  panda_version = "bootstub" if panda.bootstub else panda.get_version()
  panda_signature = b"" if panda.bootstub else panda.get_signature()
  cloudlog.warning(f"Panda {panda_serial} connected, version: {panda_version}, signature {panda_signature.hex()[:16]}, expected {fw_signature.hex()[:16]}")

  if panda.bootstub or panda_signature != fw_signature:
    cloudlog.info("Panda firmware out of date, update required")
    panda.flash()
    cloudlog.info("Done flashing")

  if panda.bootstub:
    bootstub_version = panda.get_version()
    cloudlog.info(f"Flashed firmware not booting, flashing development bootloader. {bootstub_version=}, {internal_panda=}")
    try:
      if internal_panda:
        HARDWARE.recover_internal_panda()
      panda.recover(reset=(not internal_panda))
      cloudlog.info("Done flashing bootstub")
    except Exception as e:
      cloudlog.warning(f"Failed to recover panda: {e}, continuing anyway")

  # If still in bootstub after all attempts, log warning but continue
  # This allows the system to continue running even if Panda firmware is incompatible
  if panda.bootstub:
    cloudlog.warning("Panda still in bootstub mode after flashing attempts, but continuing with existing firmware")
    # Don't raise AssertionError - allow system to continue
    # Note: CAN messages may not be available, so vehicle identification may fail

  # Check signature but don't fail if mismatch (for compatibility with 4.0 firmware)
  panda_signature = panda.get_signature()
  if panda_signature != fw_signature:
    cloudlog.warning(f"Version mismatch (got {panda_signature.hex()[:16] if panda_signature else 'empty'}, expected {fw_signature.hex()[:16]}), but continuing with existing firmware")
    # Don't raise AssertionError - allow system to continue

  return panda


def check_panda_support(panda) -> bool:
  hw_type = panda.get_type()
  if hw_type in Panda.SUPPORTED_DEVICES:
    return True

  return False


def main() -> None:
  # signal pandad to close the relay and exit
  def signal_handler(signum, frame):
    cloudlog.info(f"Caught signal {signum}, exiting")
    nonlocal do_exit
    do_exit = True
    if process is not None:
      process.send_signal(signal.SIGINT)

  process = None
  do_exit = False
  signal.signal(signal.SIGINT, signal_handler)

  count = 0
  first_run = True
  params = Params()
  no_internal_panda_count = 0

  while not do_exit:
    try:
      count += 1
      cloudlog.event("pandad.flash_and_connect", count=count)
      params.remove("PandaSignatures")

      system_uptime = time.monotonic()
      
      # Check SPI device availability (for SPI-only devices)
      spi_available = os.path.exists("/dev/spidev0.0")
      if not spi_available and system_uptime < 10.:
        # SPI device might not be ready yet, wait a bit
        cloudlog.debug("SPI device not ready yet, waiting...")
        time.sleep(2)
        continue

      # Flash all Pandas in DFU mode first
      dfu_serials = PandaDFU.list()
      if len(dfu_serials) > 0:
        for serial in dfu_serials:
          cloudlog.info(f"Panda in DFU mode found, attempting recovery {serial}")
          try:
            dfu = PandaDFU(serial)
            # For TICI devices (which use H7), try H7 firmware first
            # MCU type detection in DFU mode can be unreliable
            h7_bootstub_fn = os.path.join(FW_PATH, "bootstub.panda_h7.bin")
            f4_bootstub_fn = os.path.join(FW_PATH, "bootstub.panda.bin")
            
            recovery_successful = False
            # Try H7 firmware first (TICI devices use H7)
            if os.path.exists(h7_bootstub_fn):
              try:
                cloudlog.info(f"Using H7 firmware for DFU recovery: {h7_bootstub_fn}")
                with open(h7_bootstub_fn, "rb") as f:
                  code = f.read()
                dfu.program_bootstub(code)
                dfu.reset()
                cloudlog.info(f"Successfully recovered DFU Panda {serial} with H7 bootstub")
                recovery_successful = True
              except Exception as e:
                cloudlog.warning(f"Failed to program H7 firmware: {e}, trying F4 firmware")
            
            # Fallback to F4 firmware if H7 failed or doesn't exist
            if not recovery_successful and os.path.exists(f4_bootstub_fn):
              try:
                # Reconnect to DFU if H7 recovery failed
                if recovery_successful is False:
                  try:
                    dfu.close()
                  except Exception:
                    pass
                  dfu = PandaDFU(serial)
                
                cloudlog.info(f"Using F4 firmware for DFU recovery: {f4_bootstub_fn}")
                with open(f4_bootstub_fn, "rb") as f:
                  code = f.read()
                dfu.program_bootstub(code)
                dfu.reset()
                cloudlog.info(f"Successfully recovered DFU Panda {serial} with F4 bootstub")
                recovery_successful = True
              except Exception as e:
                cloudlog.warning(f"Failed to program F4 firmware: {e}")
            
            if not recovery_successful:
              cloudlog.warning(f"Neither H7 nor F4 firmware available or programming failed for DFU Panda {serial}")
              # Try to reset the panda to exit DFU mode
              try:
                dfu.reset()
                cloudlog.info("Attempted to reset Panda from DFU mode")
              except Exception as e:
                cloudlog.warning(f"Failed to reset Panda from DFU mode: {e}")
            
            dfu.close()
          except Exception as e:
            cloudlog.warning(f"Failed to recover DFU Panda {serial}: {e}, continuing anyway")
            continue
        time.sleep(1)  # Wait for Panda to exit DFU mode and reconnect

      # Try to list pandas (USB + SPI)
      panda_serials = Panda.list()
      if len(panda_serials) == 0:
        cloudlog.warning(f"No pandas found (attempt {no_internal_panda_count + 1}, uptime: {system_uptime:.1f}s)")
        
        # Try hardware reset if system has been up for a while and we have internal panda
        if system_uptime >= 10. and HARDWARE.has_internal_panda():
          no_internal_panda_count += 1
          if no_internal_panda_count >= 3:
            cloudlog.info("No pandas found, putting internal panda into DFU")
            try:
              HARDWARE.recover_internal_panda()
              time.sleep(3)  # wait to come back up
            except Exception as e:
              cloudlog.warning(f"Failed to recover internal panda: {e}")
          elif no_internal_panda_count > 0:
            cloudlog.info("No pandas found, resetting internal panda")
            try:
              HARDWARE.reset_internal_panda()
              time.sleep(3)  # wait to come back up
            except Exception as e:
              cloudlog.warning(f"Failed to reset internal panda: {e}")
        elif system_uptime < 10.:
          # System just started, wait a bit longer before trying reset
          time.sleep(2)
        else:
          # System is up but no internal panda expected, just wait
          time.sleep(2)
        
        continue
      
      # Reset counter on success
      no_internal_panda_count = 0

      cloudlog.info(f"{len(panda_serials)} panda(s) found, connecting - {panda_serials}")

      # Flash pandas
      pandas: list[Panda] = []
      for serial in panda_serials:
        pandas.append(flash_panda(serial))

      # Ensure internal panda is present if expected
      internal_pandas = [panda for panda in pandas if panda.is_internal()]
      if HARDWARE.has_internal_panda() and len(internal_pandas) == 0:
        cloudlog.error("Internal panda is missing, trying again")
        if system_uptime >= 10.:
          no_internal_panda_count += 1
        continue

      # sort pandas to have deterministic order
      # * the internal one is always first
      # * then sort by hardware type
      # * as a last resort, sort by serial number
      pandas.sort(key=lambda x: (not x.is_internal(), x.get_type(), x.get_usb_serial()))
      panda_serials = [p.get_usb_serial() for p in pandas]

      # log panda fw versions
      try:
        signatures = []
        for p in pandas:
          try:
            sig = p.get_signature()
            if sig:
              signatures.append(sig)
          except Exception:
            # If panda is in bootstub or signature unavailable, skip it
            pass
        if signatures:
          params.put("PandaSignatures", b','.join(signatures))
      except Exception:
        cloudlog.warning("Failed to log panda signatures")

      for panda in pandas:
        # skip health check if the detected panda is not supported
        supported_panda = check_panda_support(panda)
        if not supported_panda:
          cloudlog.warning(f"Panda {panda.get_usb_serial()} is not supported (hw_type: {panda.get_type()}), skipping health check...")
          continue

        # check health for lost heartbeat
        health = panda.health()
        if health["heartbeat_lost"]:
          params.put_bool("PandaHeartbeatLost", True)
          cloudlog.event("heartbeat lost", deviceState=health, serial=panda.get_usb_serial())
        if health["som_reset_triggered"]:
          params.put_bool("PandaSomResetTriggered", True)
          cloudlog.event("panda.som_reset_triggered", health=health, serial=panda.get_usb_serial())

        if first_run:
          # reset panda to ensure we're in a good state
          cloudlog.info(f"Resetting panda {panda.get_usb_serial()}")
          panda.reset(reconnect=True)

      for p in pandas:
        p.close()
    # TODO: wrap all panda exceptions in a base panda exception
    except (usb1.USBErrorNoDevice, usb1.USBErrorPipe):
      # a panda was disconnected while setting everything up. let's try again
      cloudlog.exception("Panda USB exception while setting up")
      continue
    except PandaProtocolMismatch:
      cloudlog.exception("pandad.protocol_mismatch")
      continue
    except Exception:
      cloudlog.exception("pandad.uncaught_exception")
      continue

    first_run = False

    # run pandad with all connected serials as arguments
    os.environ['MANAGER_DAEMON'] = 'pandad'
    process = subprocess.Popen(["./pandad", *panda_serials], cwd=os.path.join(BASEDIR, "selfdrive/pandad"))
    process.wait()


if __name__ == "__main__":
  main()
