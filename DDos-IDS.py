import joblib
import pandas as pd
import numpy as np
import subprocess
import os
import sys
import time
import argparse
import logging
import threading
import queue

# ─── LOGGING CONFIGURATION ─────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("ids_alerts.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─── ARGPARSE CONFIGURATION ────────────────────────────
parser = argparse.ArgumentParser(description="FYP DDoS Detection System")
parser.add_argument("-i", "--interface", type=str, default="eth0", help="Network interface to monitor (default: eth0)")
parser.add_argument("-t", "--time", type=int, default=10, help="Capture time per cycle in seconds (default: 10)")
parser.add_argument("-m", "--model_dir", type=str, default="./", help="Directory containing ML models")
parser.add_argument("-p", "--pcap_dir", type=str, default="/dev/shm", help="Directory to store temporary pcap files (default: /dev/shm)")
args = parser.parse_args()

INTERFACE = args.interface
CAPTURE_TIME = args.time
MODEL_PATH = args.model_dir
PCAP_DIR = args.pcap_dir

# Global queue for inter-thread communication (Bounded to prevent memory leaks/processing lag)
analysis_queue = queue.Queue(maxsize=2)

# ─── LOAD MODELS ──────────────────────────────────────
try:
    logger.info("Loading models...")
    rf       = joblib.load(os.path.join(MODEL_PATH, 'random_forest.pkl'))
    xgb      = joblib.load(os.path.join(MODEL_PATH, 'xgboost.pkl'))
    iso      = joblib.load(os.path.join(MODEL_PATH, 'isolation_forest.pkl'))
    scaler   = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
    features = joblib.load(os.path.join(MODEL_PATH, 'feature_names.pkl'))
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    exit(1)

def capture_traffic(cycle):
    """Capture live traffic for CAPTURE_TIME seconds and save to a unique file."""
    pcap_file = os.path.join(PCAP_DIR, f"capture_{cycle}.pcap")
    logger.info(f"Capturing traffic on {INTERFACE} for {CAPTURE_TIME}s (Cycle {cycle})...")

    try:
        if os.path.exists(pcap_file):
            os.remove(pcap_file)

        proc = subprocess.Popen([
            'tcpdump', '-i', INTERFACE,
            '-w', pcap_file,
            '-B', '8192',  # Allocates an 8MB buffer to stop kernel drops
            '--immediate-mode'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        time.sleep(CAPTURE_TIME)
        proc.terminate()
        proc.wait()

        return pcap_file
    except Exception as e:
        logger.error(f"Capture error: {e}")
        return None

def pcap_to_flows(pcap_path):
    """Convert pcap to flow features using CICFlowMeter"""
    csv_path = pcap_path.replace('.pcap', '.csv')
    logger.debug(f"Converting packets to flows from {pcap_path} using CICFlowMeter...")
    try:
        if not os.path.exists(pcap_path):
            logger.warning(f"pcap file not found: {pcap_path}")
            return None

        # Run cicflowmeter as a subprocess using its absolute path from the venv
        cic_bin = os.path.join(os.path.dirname(sys.executable), 'cicflowmeter')
        proc = subprocess.Popen([
            cic_bin, '-f', pcap_path, '-c', csv_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.wait()

        if not os.path.exists(csv_path):
            logger.warning(f"CICFlowMeter failed to generate CSV for {pcap_path}")
            return None

        # Load the generated CSV
        df = pd.read_csv(csv_path)

        # Clean up the CSV file
        try:
            os.remove(csv_path)
        except:
            pass

        if df is None or len(df) == 0:
            logger.warning(f"No flows extracted from {pcap_path}")
            return None

        return df

    except Exception as e:
        logger.error(f"Flow extraction error: {e}")
        return None

def preprocess_flows(df):
    """Format CICFlowMeter output to match exactly the training features"""
    # CICFlowMeter python outputs columns in lowercase, but the dataset is capitalized/spaces.
    # We must normalize the column names to match the expected format.

    # First, strip leading/trailing spaces
    df.columns = df.columns.str.strip()

    # CICFlowMeter-py often uses lowercase with underscores, e.g., 'flow_duration'
    # We map them to the CIC-DDoS2019 format.
    col_map = {
        'flow_duration': 'Flow Duration',
        'tot_fwd_pkts': 'Total Fwd Packets',
        'tot_bwd_pkts': 'Total Backward Packets',
        'totlen_fwd_pkts': 'Total Length of Fwd Packets',
        'totlen_bwd_pkts': 'Total Length of Bwd Packets',
        'fwd_pkt_len_max': 'Fwd Packet Length Max',
        'fwd_pkt_len_min': 'Fwd Packet Length Min',
        'fwd_pkt_len_mean': 'Fwd Packet Length Mean',
        'fwd_pkt_len_std': 'Fwd Packet Length Std',
        'bwd_pkt_len_max': 'Bwd Packet Length Max',
        'bwd_pkt_len_min': 'Bwd Packet Length Min',
        'bwd_pkt_len_mean': 'Bwd Packet Length Mean',
        'bwd_pkt_len_std': 'Bwd Packet Length Std',
        'flow_byts_s': 'Flow Bytes/s',
        'flow_pkts_s': 'Flow Packets/s',
        'flow_iat_mean': 'Flow IAT Mean',
        'flow_iat_std': 'Flow IAT Std',
        'flow_iat_max': 'Flow IAT Max',
        'flow_iat_min': 'Flow IAT Min',
        'fwd_iat_tot': 'Fwd IAT Total',
        'fwd_iat_mean': 'Fwd IAT Mean',
        'fwd_iat_std': 'Fwd IAT Std',
        'fwd_iat_max': 'Fwd IAT Max',
        'fwd_iat_min': 'Fwd IAT Min',
        'bwd_iat_tot': 'Bwd IAT Total',
        'bwd_iat_mean': 'Bwd IAT Mean',
        'bwd_iat_std': 'Bwd IAT Std',
        'bwd_iat_max': 'Bwd IAT Max',
        'bwd_iat_min': 'Bwd IAT Min',
        'fwd_psh_flags': 'Fwd PSH Flags',
        'bwd_psh_flags': 'Bwd PSH Flags',
        'fwd_urg_flags': 'Fwd URG Flags',
        'bwd_urg_flags': 'Bwd URG Flags',
        'fwd_header_len': 'Fwd Header Length',
        'bwd_header_len': 'Bwd Header Length',
        'fwd_pkts_s': 'Fwd Packets/s',
        'bwd_pkts_s': 'Bwd Packets/s',
        'pkt_len_min': 'Min Packet Length',
        'pkt_len_max': 'Max Packet Length',
        'pkt_len_mean': 'Packet Length Mean',
        'pkt_len_std': 'Packet Length Std',
        'pkt_len_var': 'Packet Length Variance',
        'fin_flag_cnt': 'FIN Flag Count',
        'syn_flag_cnt': 'SYN Flag Count',
        'rst_flag_cnt': 'RST Flag Count',
        'psh_flag_cnt': 'PSH Flag Count',
        'ack_flag_cnt': 'ACK Flag Count',
        'urg_flag_cnt': 'URG Flag Count',
        'cwe_flag_count': 'CWE Flag Count',
        'ece_flag_cnt': 'ECE Flag Count',
        'down_up_ratio': 'Down/Up Ratio',
        'pkt_size_avg': 'Average Packet Size',
        'fwd_seg_size_avg': 'Avg Fwd Segment Size',
        'bwd_seg_size_avg': 'Avg Bwd Segment Size',
        'fwd_header_len': 'Fwd Header Length.1', # Some datasets map this twice
        'fwd_byts_b_avg': 'Fwd Avg Bytes/Bulk',
        'fwd_pkts_b_avg': 'Fwd Avg Packets/Bulk',
        'fwd_blk_rate_avg': 'Fwd Avg Bulk Rate',
        'bwd_byts_b_avg': 'Bwd Avg Bytes/Bulk',
        'bwd_pkts_b_avg': 'Bwd Avg Packets/Bulk',
        'bwd_blk_rate_avg': 'Bwd Avg Bulk Rate',
        'subflow_fwd_pkts': 'Subflow Fwd Packets',
        'subflow_fwd_byts': 'Subflow Fwd Bytes',
        'subflow_bwd_pkts': 'Subflow Bwd Packets',
        'subflow_bwd_byts': 'Subflow Bwd Bytes',
        'init_fwd_win_byts': 'Init_Win_bytes_forward',
        'init_bwd_win_byts': 'Init_Win_bytes_backward',
        'fwd_act_data_pkts': 'act_data_pkt_fwd',
        'fwd_seg_size_min': 'min_seg_size_forward',
        'active_mean': 'Active Mean',
        'active_std': 'Active Std',
        'active_max': 'Active Max',
        'active_min': 'Active Min',
        'idle_mean': 'Idle Mean',
        'idle_std': 'Idle Std',
        'idle_max': 'Idle Max',
        'idle_min': 'Idle Min'
    }

    # Rename columns if they are in the lowercase format
    df.rename(columns=col_map, inplace=True)

    # Capitalize 'protocol' if it didn't get mapped
    if 'protocol' in df.columns:
        df.rename(columns={'protocol': 'Protocol'}, inplace=True)

    # Build final dataframe in exact training feature order
    final_df = pd.DataFrame()
    for col in features:
        if col in df.columns:
            # Convert values to numeric, coercing errors to NaN
            final_df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            final_df[col] = 0

    # Handle infinities and NaN
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.fillna(0, inplace=True)

    return final_df

def detect(df, raw_df, cycle):
    """Run protocol rule-based filter then ML models"""
    if df is None or len(df) == 0:
        logger.warning(f"Cycle {cycle}: No flows to analyse")
        return

    total = len(df)

    # ── Protocol Rule-Based Detection ────────────────────────
    # Try to find SYN and Duration columns whether they are lowercase or mapped
    syn_col = 'syn_flag_cnt' if 'syn_flag_cnt' in raw_df.columns else 'SYN Flag Count'
    dur_col = 'flow_duration' if 'flow_duration' in raw_df.columns else 'Flow Duration'
    src_col = 'src_ip' if 'src_ip' in raw_df.columns else ('Source IP' if 'Source IP' in raw_df.columns else None)

    syn_ratio = (raw_df[syn_col] >= 1).mean() if syn_col in raw_df.columns else 0
    zero_dur_ratio = (raw_df[dur_col] == 0).mean() if dur_col in raw_df.columns else 0
    single_src = raw_df[src_col].nunique() if src_col else 0

    # Packets could be mapped to 'Total Fwd Packets' and 'Total Backward Packets'
    fwd_pkts = 'tot_fwd_pkts' if 'tot_fwd_pkts' in raw_df.columns else 'Total Fwd Packets'
    bwd_pkts = 'tot_bwd_pkts' if 'tot_bwd_pkts' in raw_df.columns else 'Total Backward Packets'

    if fwd_pkts in raw_df.columns and bwd_pkts in raw_df.columns:
        # Convert to numeric to avoid string concatenation issues
        total_packets = pd.to_numeric(raw_df[fwd_pkts], errors='coerce').fillna(0).sum() + pd.to_numeric(raw_df[bwd_pkts], errors='coerce').fillna(0).sum()
    else:
        total_packets = 0

    pkt_per_flow = total_packets / total if total > 0 else 0

    rule_ddos = False
    rule_reason = ""

    if syn_ratio >= 0.25 and zero_dur_ratio >= 0.25 and total > 1000:
        rule_ddos = True
        rule_reason = f"SYN flood (SYN: {syn_ratio:.1%}, zero-dur: {zero_dur_ratio:.1%})"

    logger.info(f"Cycle {cycle} Stats | SYN: {syn_ratio:.1%} | Zero-Dur: {zero_dur_ratio:.1%} | Srcs: {single_src} | Flows: {total:,} | Pkts/Flow: {pkt_per_flow:.1f}")

    if rule_ddos:
        logger.warning(f"Cycle {cycle}: Rule triggered: {rule_reason}")

    # ── ML Models ───────────────────────────────────────
    X = scaler.transform(df)
    rf_preds  = rf.predict(X)
    xgb_preds = xgb.predict(X)
    iso_raw   = iso.predict(X)
    iso_preds = [1 if x == -1 else 0 for x in iso_raw]

    rf_ratio  = int(sum(rf_preds)) / total
    xgb_ratio = int(sum(xgb_preds)) / total
    iso_ratio = int(sum(iso_preds)) / total

    min_flows = 20
    enough_data = total >= min_flows

    supervised_alert = rf_ratio >= 0.30 and xgb_ratio >= 0.30 and enough_data
    high_confidence = rf_ratio >= 0.60 and xgb_ratio >= 0.60 and enough_data
    
    logger.info(f"Cycle {cycle} ML Flags | RF: {rf_ratio:.1%} | XGB: {xgb_ratio:.1%} | ISO: {iso_ratio:.1%}")

    # ── Final Verdict ────────────────────────────────────
    if rule_ddos:
        logger.critical(f"Cycle {cycle} 🚨 FINAL VERDICT: DDoS ATTACK DETECTED! (Protocol Anomaly — {rule_reason})")
    elif supervised_alert and rule_ddos:
        logger.critical(f"Cycle {cycle} 🚨 FINAL VERDICT: DDoS ATTACK DETECTED! (Rule + ML consensus: {rf_ratio:.1%} of flows flagged)")
    elif high_confidence and enough_data:
        logger.critical(f"Cycle {cycle} 🚨 FINAL VERDICT: DDoS ATTACK DETECTED! (High ML confidence: {rf_ratio:.1%} of flows flagged)")
    elif supervised_alert and not rule_ddos and enough_data:
        logger.warning(f"Cycle {cycle} ⚠  FINAL VERDICT: SUSPICIOUS TRAFFIC (ML flags {rf_ratio:.1%} of {total} flows)")
    else:
        if not enough_data:
            logger.info(f"Cycle {cycle} ✅ FINAL VERDICT: BENIGN (Insufficient flows: {total})")
        else:
            logger.info(f"Cycle {cycle} ✅ FINAL VERDICT: BENIGN (Within normal range)")

def analysis_worker():
    """Worker thread that processes pcaps from the queue."""
    while True:
        cycle, pcap_path = analysis_queue.get()
        if pcap_path is None:  # Sentinel value to exit
            break

        logger.info(f"Starting analysis for Cycle {cycle}")
        try:
            raw_df = pcap_to_flows(pcap_path)
            if raw_df is not None:
                df = preprocess_flows(raw_df.copy())
                detect(df, raw_df, cycle)
        except Exception as e:
            logger.error(f"Error during analysis of Cycle {cycle}: {e}")
        finally:
            # Clean up pcap file after analysis
            if pcap_path and os.path.exists(pcap_path):
                try:
                    os.remove(pcap_path)
                except Exception as e:
                    logger.warning(f"Failed to delete {pcap_path}: {e}")
            analysis_queue.task_done()

# ─── MAIN ──────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("="*55)
    logger.info("  FYP DDoS Detection System — TP077433")
    logger.info(f"  Interface: {INTERFACE}")
    logger.info(f"  Cycle Duration: {CAPTURE_TIME} seconds")
    logger.info("="*55)

    # Start analysis thread
    analyzer_thread = threading.Thread(target=analysis_worker, daemon=True)
    analyzer_thread.start()

    cycle = 1
    try:
        while True:
            pcap_path = capture_traffic(cycle)
            if pcap_path:
                try:
                    # Try to put the file in the queue without blocking
                    analysis_queue.put_nowait((cycle, pcap_path))
                except queue.Full:
                    # LOAD SHEDDING: If the analyzer is too far behind, drop the capture to stay real-time
                    logger.warning(f"Queue full! Analyzer is lagging. Dropping Cycle {cycle} to maintain real-time monitoring.")
                    try:
                        os.remove(pcap_path)
                    except Exception as e:
                        logger.error(f"Failed to delete dropped pcap {pcap_path}: {e}")
            cycle += 1
    except KeyboardInterrupt:
        logger.info("Detection system stopped by user")
    except Exception as e:
        logger.error(f"Main loop error: {e}")
    finally:
        # Signal the analyzer thread to stop and wait for it
        analysis_queue.put((None, None))
        analyzer_thread.join(timeout=5)
        logger.info("System shutdown complete")
