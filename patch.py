with open("DDos-IDS.py", "r") as f:
    content = f.read()

# Fix the safe division approach to avoid RuntimeWarnings
search1 = """    # Safe divisions using np.where to avoid divide by zero errors/warnings
    df['Flow Bytes/s'] = np.where(duration_s > 0, (df['Total Length of Fwd Packets'] + df['Total Length of Bwd Packets']) / duration_s, 0)
    df['Flow Packets/s'] = np.where(duration_s > 0, (df['Total Fwd Packets'] + df['Total Backward Packets']) / duration_s, 0)
    df['Fwd Packets/s'] = np.where(duration_s > 0, df['Total Fwd Packets'] / duration_s, 0)
    df['Bwd Packets/s'] = np.where(duration_s > 0, df['Total Backward Packets'] / duration_s, 0)"""

replace1 = """    # Safe divisions to avoid divide by zero errors/warnings
    safe_duration = duration_s.replace(0, np.nan)
    df['Flow Bytes/s'] = ((df['Total Length of Fwd Packets'] + df['Total Length of Bwd Packets']) / safe_duration).fillna(0)
    df['Flow Packets/s'] = ((df['Total Fwd Packets'] + df['Total Backward Packets']) / safe_duration).fillna(0)
    df['Fwd Packets/s'] = (df['Total Fwd Packets'] / safe_duration).fillna(0)
    df['Bwd Packets/s'] = (df['Total Backward Packets'] / safe_duration).fillna(0)"""

search2 = """    total_pkts = df['Total Fwd Packets'] + df['Total Backward Packets']
    df['Average Packet Size'] = np.where(total_pkts > 0, (df['Total Length of Fwd Packets'] + df['Total Length of Bwd Packets']) / total_pkts, 0)"""

replace2 = """    total_pkts = df['Total Fwd Packets'] + df['Total Backward Packets']
    safe_total_pkts = total_pkts.replace(0, np.nan)
    df['Average Packet Size'] = ((df['Total Length of Fwd Packets'] + df['Total Length of Bwd Packets']) / safe_total_pkts).fillna(0)"""

search3 = """    df['Down/Up Ratio'] = np.where(df['Total Fwd Packets'] > 0, df['Total Backward Packets'] / df['Total Fwd Packets'], 0)"""

replace3 = """    safe_fwd_pkts = df['Total Fwd Packets'].replace(0, np.nan)
    df['Down/Up Ratio'] = (df['Total Backward Packets'] / safe_fwd_pkts).fillna(0)"""

content = content.replace(search1, replace1)
content = content.replace(search2, replace2)
content = content.replace(search3, replace3)

with open("DDos-IDS.py", "w") as f:
    f.write(content)
