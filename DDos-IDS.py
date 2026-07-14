import joblib
import pandas as pd
import numpy as np
import subprocess
import os
import time
import argparse
import logging
import threading
import queue
from nfstream import NFStreamer

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
    """Convert pcap to flow features using NFStream"""
    logger.debug(f"Converting packets to flows from {pcap_path}...")
    try:
        if not os.path.exists(pcap_path):
            logger.warning(f"pcap file not found: {pcap_path}")
            return None

        streamer = NFStreamer(
            source=pcap_path,
            statistical_analysis=True,
            splt_analysis=0
        )

        df = streamer.to_pandas()

        if df is None or len(df) == 0:
            logger.warning(f"No flows extracted from {pcap_path}")
            return None

        return df

    except Exception as e:
        logger.error(f"Flow extraction error: {e}")
        return None

def preprocess_flows(df):
    """Map NFStream columns to match training features"""
    col_map = {
        'protocol':                      'Protocol',
        'bidirectional_duration_ms':     'Flow Duration',
        'src2dst_packets':               'Total Fwd Packets',
        'dst2src_packets':               'Total Backward Packets',
        'src2dst_bytes':                 'Total Length of Fwd Packets',
        'dst2src_bytes':                 'Total Length of Bwd Packets',
        'src2dst_max_ps':                'Fwd Packet Length Max',
        'src2dst_min_ps':                'Fwd Packet Length Min',
        'src2dst_mean_ps':               'Fwd Packet Length Mean',
        'src2dst_stddev_ps':             'Fwd Packet Length Std',
        'dst2src_max_ps':                'Bwd Packet Length Max',
        'dst2src_min_ps':                'Bwd Packet Length Min',
        'dst2src_mean_ps':               'Bwd Packet Length Mean',
        'dst2src_stddev_ps':             'Bwd Packet Length Std',
        'bidirectional_mean_ps':         'Packet Length Mean',
        'bidirectional_stddev_ps':       'Packet Length Std',
        'bidirectional_max_ps':          'Max Packet Length',
        'bidirectional_min_ps':          'Min Packet Length',
        'src2dst_mean_piat_ms':          'Fwd IAT Mean',
        'src2dst_stddev_piat_ms':        'Fwd IAT Std',
        'src2dst_max_piat_ms':           'Fwd IAT Max',
        'src2dst_min_piat_ms':           'Fwd IAT Min',
        'dst2src_mean_piat_ms':          'Bwd IAT Mean',
        'dst2src_stddev_piat_ms':        'Bwd IAT Std',
        'dst2src_max_piat_ms':           'Bwd IAT Max',
        'dst2src_min_piat_ms':           'Bwd IAT Min',
        'bidirectional_mean_piat_ms':    'Flow IAT Mean',
        'bidirectional_stddev_piat_ms':  'Flow IAT Std',
        'bidirectional_max_piat_ms':     'Flow IAT Max',
        'bidirectional_min_piat_ms':     'Flow IAT Min',
        'bidirectional_syn_packets':     'SYN Flag Count',
        'bidirectional_fin_packets':     'FIN Flag Count',
        'bidirectional_rst_packets':     'RST Flag Count',
        'bidirectional_psh_packets':     'PSH Flag Count',
        'bidirectional_ack_packets':     'ACK Flag Count',
        'bidirectional_urg_packets':     'URG Flag Count',
        'bidirectional_ece_packets':     'ECE Flag Count',
        'bidirectional_cwr_packets':     'CWE Flag Count',
        'src2dst_syn_packets':           'Fwd PSH Flags',
        'src2dst_fin_packets':           'Fwd URG Flags',
        'dst2src_syn_packets':           'Bwd PSH Flags',
        'dst2src_fin_packets':           'Bwd URG Flags',
        'src2dst_duration_ms':           'Fwd IAT Total',
        'dst2src_duration_ms':           'Bwd IAT Total',
        'bidirectional_packets':         'Subflow Fwd Packets',
        'bidirectional_bytes':           'Subflow Fwd Bytes',
    }

    df = df.rename(columns=col_map)
    duration_s = df['Flow Duration'] / 1000.0  # ms to seconds

    # Safe divisions to avoid divide by zero errors/warnings
    safe_duration = duration_s.replace(0, np.nan)
    df['Flow Bytes/s'] = ((df['Total Length of Fwd Packets'] + df['Total Length of Bwd Packets']) / safe_duration).fillna(0)
    df['Flow Packets/s'] = ((df['Total Fwd Packets'] + df['Total Backward Packets']) / safe_duration).fillna(0)
    df['Fwd Packets/s'] = (df['Total Fwd Packets'] / safe_duration).fillna(0)
    df['Bwd Packets/s'] = (df['Total Backward Packets'] / safe_duration).fillna(0)

    df['Packet Length Variance'] = df['Packet Length Std'] ** 2

    total_pkts = df['Total Fwd Packets'] + df['Total Backward Packets']
    safe_total_pkts = total_pkts.replace(0, np.nan)
    df['Average Packet Size'] = ((df['Total Length of Fwd Packets'] + df['Total Length of Bwd Packets']) / safe_total_pkts).fillna(0)

    df['Avg Fwd Segment Size'] = df['Fwd Packet Length Mean']
    df['Avg Bwd Segment Size'] = df['Bwd Packet Length Mean']
    df['Fwd Header Length']   = df['Total Fwd Packets'] * 20
    df['Bwd Header Length']   = df['Total Backward Packets'] * 20
    df['Fwd Header Length.1'] = df['Fwd Header Length']
    df['Subflow Bwd Packets'] = df['Total Backward Packets']
    df['Subflow Bwd Bytes']   = df['Total Length of Bwd Packets']

    safe_fwd_pkts = df['Total Fwd Packets'].replace(0, np.nan)
    df['Down/Up Ratio'] = (df['Total Backward Packets'] / safe_fwd_pkts).fillna(0)

    # Bulk features — set to 0 (not available in NFStream)
    missing_cols = ['Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
                    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
                    'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
                    'act_data_pkt_fwd', 'min_seg_size_forward',
                    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
                    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Inbound']

    # Efficiently add missing columns
    missing_df = pd.DataFrame(0, index=df.index, columns=missing_cols)
    df = pd.concat([df, missing_df], axis=1)

    final_df = pd.DataFrame()
    for col in features:
        if col in df.columns:
            final_df[col] = df[col].values
        else:
            final_df[col] = 0

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
    syn_ratio = (raw_df['bidirectional_syn_packets'] == 1).mean() if 'bidirectional_syn_packets' in raw_df.columns else 0
    zero_dur_ratio = (raw_df['bidirectional_duration_ms'] == 0).mean() if 'bidirectional_duration_ms' in raw_df.columns else 0
    single_src = raw_df['src_ip'].nunique() if 'src_ip' in raw_df.columns else 0
    total_packets = raw_df['bidirectional_packets'].sum() if 'bidirectional_packets' in raw_df.columns else 0
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
