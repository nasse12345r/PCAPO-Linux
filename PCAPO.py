import joblib
import pandas as pd
import numpy as np
import subprocess
import os
import time
from nfstream import NFStreamer

# ─── CONFIG ───────────────────────────────────────────
MODEL_PATH   = '/home/DS/Desktop/PCAPO/models_nfs/'
PCAP_FILE    = '/tmp/capture.pcap'
INTERFACE    = 'eth0'
CAPTURE_TIME = 30
MIN_FLOWS    = 20
# ──────────────────────────────────────────────────────

# Load models
print("[*] Loading NFStream-native models...")
rf      = joblib.load(MODEL_PATH + 'rf_nfs.pkl')
xgb     = joblib.load(MODEL_PATH + 'xgb_nfs.pkl')
iso     = joblib.load(MODEL_PATH + 'iso_nfs.pkl')
scaler  = joblib.load(MODEL_PATH + 'scaler_nfs.pkl')
features = joblib.load(MODEL_PATH + 'nfs_features.pkl')
print(f"[+] Models loaded — {len(features)} features\n")

def capture_traffic():
    """Capture live traffic"""
    print(f"[*] Capturing on {INTERFACE} for {CAPTURE_TIME}s...")
    try:
        if os.path.exists(PCAP_FILE):
            os.remove(PCAP_FILE)

        proc = subprocess.Popen([
            'tcpdump', '-i', INTERFACE,
            '-w', PCAP_FILE,
            '--immediate-mode'
        ])
        time.sleep(CAPTURE_TIME)
        proc.terminate()
        proc.wait()
        print("[+] Capture complete")
        return True
    except Exception as e:
        print(f"[-] Capture error: {e}")
        return False

def pcap_to_flows():
    """Convert pcap to NFStream flows"""
    print("[*] Extracting flows...")
    try:
        if not os.path.exists(PCAP_FILE):
            print("[-] No pcap file found")
            return None

        streamer = NFStreamer(
            source=PCAP_FILE,
            statistical_analysis=True,
            splt_analysis=0
        )
        df = streamer.to_pandas()

        if df is None or len(df) == 0:
            print("[-] No flows extracted")
            return None

        print(f"[+] Extracted {len(df)} flows")
        return df

    except Exception as e:
        print(f"[-] Flow error: {e}")
        return None

def preprocess(df):
    """Extract exactly the features the models were trained on"""
    final = pd.DataFrame()
    for col in features:
        if col in df.columns:
            final[col] = df[col].values
        else:
            final[col] = 0

    final.replace([np.inf, -np.inf], np.nan, inplace=True)
    final.fillna(0, inplace=True)
    return final

def detect(df, raw_df):
    """Run RF, XGBoost and Isolation Forest"""
    total = len(df)

    if total < MIN_FLOWS:
        print(f"[!] Only {total} flows — insufficient for reliable analysis\n")
        return

    print(f"[*] Analysing {total} flows...")

    # Scale
    X = scaler.transform(df)

    # Supervised models
    rf_preds  = rf.predict(X)
    xgb_preds = xgb.predict(X)

    # Isolation Forest
    iso_raw   = iso.predict(X)
    iso_preds = [1 if x == -1 else 0 for x in iso_raw]

    rf_ddos  = int(sum(rf_preds))
    xgb_ddos = int(sum(xgb_preds))
    iso_ddos = int(sum(iso_preds))

    rf_ratio  = rf_ddos / total
    xgb_ratio = xgb_ddos / total
    iso_ratio = iso_ddos / total

    # ── Traffic Statistics ─────────────────────────────
    total_pkts  = int(raw_df['bidirectional_packets'].sum()) \
                  if 'bidirectional_packets' in raw_df.columns else 0
    unique_src  = raw_df['src_ip'].nunique() \
                  if 'src_ip' in raw_df.columns else 0
    syn_ratio   = float((raw_df['bidirectional_syn_packets'] >= 1).mean()) \
                  if 'bidirectional_syn_packets' in raw_df.columns else 0

    # ── UDP Flood Fingerprint Detection ───────────────
    # UDP flood has extremely uniform packet counts and duration
    pkt_std     = float(raw_df['bidirectional_packets'].std()) \
                  if 'bidirectional_packets' in raw_df.columns else 99
    dur_std     = float(raw_df['bidirectional_duration_ms'].std()) \
                  if 'bidirectional_duration_ms' in raw_df.columns else 99
    pkt_mean    = float(raw_df['bidirectional_packets'].mean()) \
                  if 'bidirectional_packets' in raw_df.columns else 0

    # Attack signature: uniform flows + high volume
    udp_flood = (
        pkt_std < 2.0 and          # very uniform packet count
        dur_std < 2000 and          # very uniform duration
        total > 1000 and            # high flow count
        pkt_mean > 10               # not just single-packet flows
    )

    # SYN flood signature
    syn_flood = (
        syn_ratio >= 0.25 and
        total > 1000
    )

    print("\n" + "="*55)
    print(f"  DETECTION RESULTS — {total} flows | {total_pkts:,} packets")
    print("="*55)

    print(f"\n  Traffic Statistics:")
    print(f"    Unique source IPs  : {unique_src}")
    print(f"    SYN-only flows     : {syn_ratio:.1%}")
    print(f"    Packets/flow mean  : {pkt_mean:.1f}")
    print(f"    Packets/flow std   : {pkt_std:.2f}")
    print(f"    Duration std (ms)  : {dur_std:.1f}")

    if udp_flood:
        print(f"\n  [!] UDP Flood Fingerprint: uniform flows detected")
        print(f"      pkt_std={pkt_std:.2f}, dur_std={dur_std:.0f}")
    if syn_flood:
        print(f"\n  [!] SYN Flood Fingerprint: high SYN ratio detected")

    print(f"\n  Supervised Models (Known DDoS Detection):")
    print(f"    [RF]  BENIGN: {total-rf_ddos:<6} DDoS: {rf_ddos} ({rf_ratio:.1%})")
    if rf_ddos > 0:
        print(f"          ⚠  Random Forest flagged {rf_ddos} DDoS flows")

    print(f"    [XGB] BENIGN: {total-xgb_ddos:<6} DDoS: {xgb_ddos} ({xgb_ratio:.1%})")
    if xgb_ddos > 0:
        print(f"          ⚠  XGBoost flagged {xgb_ddos} DDoS flows")

    print(f"\n  Anomaly Detection (Unknown Attack Detection):")
    print(f"    [ISO] NORMAL: {total-iso_ddos:<6} Anomaly: {iso_ddos} ({iso_ratio:.1%})")
    if iso_ddos > 0:
        print(f"          ⚠  Isolation Forest flagged {iso_ddos} anomalous flows")

    # ── Final Verdict ──────────────────────────────────
    print("\n" + "-"*55)

    known_ddos   = rf_ratio >= 0.30 and xgb_ratio >= 0.30
    anomaly_ddos = iso_ratio >= 0.40 and total >= MIN_FLOWS
    high_conf    = rf_ratio >= 0.60 and xgb_ratio >= 0.60
    rule_attack  = udp_flood or syn_flood

    if rule_attack and known_ddos:
        print("  🚨 KNOWN DDoS ATTACK DETECTED!")
        print(f"     Rule + ML consensus confirmed")
        if udp_flood:
            print("     → UDP Flood attack pattern")
        if syn_flood:
            print("     → SYN Flood attack pattern")

    elif rule_attack and anomaly_ddos:
        print("  🚨 DDoS ATTACK DETECTED — ANOMALOUS PATTERN!")
        print("     Rule-based + Isolation Forest both triggered")
        if udp_flood:
            print("     → Possible UDP-based unknown attack variant")
        if syn_flood:
            print("     → Possible SYN-based unknown attack variant")

    elif rule_attack:
        print("  🚨 DDoS ATTACK DETECTED!")
        if udp_flood:
            print("     → UDP Flood: uniform flow fingerprint confirmed")
            print(f"       pkt_std={pkt_std:.2f} dur_std={dur_std:.0f}ms")
        if syn_flood:
            print(f"     → SYN Flood: {syn_ratio:.1%} SYN flows")

    elif known_ddos and high_conf:
        print("  🚨 KNOWN DDoS ATTACK DETECTED!")
        print(f"     High ML confidence: RF {rf_ratio:.1%} | XGB {xgb_ratio:.1%}")

    elif known_ddos:
        print("  🚨 LIKELY DDoS ATTACK DETECTED!")
        print(f"     RF: {rf_ratio:.1%} | XGB: {xgb_ratio:.1%} flagged")

    elif anomaly_ddos and rule_attack:
        print("  ⚠  UNKNOWN/ANOMALY ATTACK DETECTED!")
        print(f"     Isolation Forest: {iso_ratio:.1%} anomalous flows")
        print("     → Pattern not matching known DDoS — possible new attack type")

    elif iso_ratio >= 0.80 and total > 1000:
        # Only flag ISO alone when extremely high ratio AND high volume
        print("  ⚠  UNKNOWN/ANOMALY ATTACK DETECTED!")
        print(f"     Isolation Forest: {iso_ratio:.1%} anomalous flows")
        print("     → Pattern not matching known DDoS — possible new attack type")

    else:
        print("  ✅ Traffic appears BENIGN")
        print(f"     RF: {rf_ratio:.1%} | XGB: {xgb_ratio:.1%} | ISO: {iso_ratio:.1%}")

    print("="*55 + "\n")

# ─── MAIN LOOP ────────────────────────────────────────
print("="*55)
print("  FYP Hybrid DDoS Detection System")
print("  Student: TP077433")
print(f"  Interface: {INTERFACE} | Cycle: {CAPTURE_TIME}s")
print("="*55 + "\n")

cycle = 1
while True:
    print(f"─── Cycle {cycle} " + "─"*38)
    try:
        if capture_traffic():
            raw_df = pcap_to_flows()
            if raw_df is not None:
                df = preprocess(raw_df.copy())
                detect(df, raw_df)
    except KeyboardInterrupt:
        print("\n[*] Detection system stopped")
        break
    except Exception as e:
        print(f"[-] Error: {e}")
    cycle += 1
    time.sleep(2)
