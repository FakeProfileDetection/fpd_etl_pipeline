#!/usr/bin/env python3
"""
Analyze and compare keystroke capture data from JavaScript and WASM implementations.
Focuses on detecting out-of-order press/release events.
"""

import csv
import sys
from collections import defaultdict
from datetime import datetime

def load_csv(filename):
    """Load CSV file and return list of events."""
    events = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append({
                'type': row['Press or Release'],
                'key': row['Key'],
                'time': int(row['Time'])
            })
    return events

def analyze_key_order(events, name):
    """Analyze keystroke order issues in the data."""
    print(f"\n{'='*60}")
    print(f"Analysis for: {name}")
    print(f"{'='*60}")
    
    # Track active keys (pressed but not released)
    active_keys = {}
    issues = []
    key_stats = defaultdict(lambda: {'presses': 0, 'releases': 0, 'issues': 0})
    
    # Skip shift-related keys for this analysis
    skip_keys = {'Key.shift', 'Key.ctrl', 'Key.alt', 'Key.cmd', 'Key.caps_lock'}
    
    for i, event in enumerate(events):
        key = event['key']
        event_type = event['type']
        time = event['time']
        
        # Update statistics
        if event_type == 'P':
            key_stats[key]['presses'] += 1
        else:
            key_stats[key]['releases'] += 1
        
        # Skip modifier keys
        if key in skip_keys:
            continue
            
        if event_type == 'P':
            if key in active_keys:
                # Key pressed again before release
                issues.append({
                    'type': 'double_press',
                    'key': key,
                    'first_press': active_keys[key],
                    'second_press': time,
                    'index': i
                })
                key_stats[key]['issues'] += 1
            active_keys[key] = time
            
        elif event_type == 'R':
            if key not in active_keys:
                # Key released without press
                issues.append({
                    'type': 'orphan_release',
                    'key': key,
                    'time': time,
                    'index': i
                })
                key_stats[key]['issues'] += 1
            else:
                # Calculate hold time
                hold_time = time - active_keys[key]
                if hold_time < 0:
                    issues.append({
                        'type': 'negative_hold_time',
                        'key': key,
                        'press_time': active_keys[key],
                        'release_time': time,
                        'hold_time': hold_time,
                        'index': i
                    })
                    key_stats[key]['issues'] += 1
                del active_keys[key]
    
    # Report findings
    print(f"\nTotal events: {len(events)}")
    print(f"Unique keys: {len(key_stats)}")
    print(f"Total issues found: {len(issues)}")
    
    # Unreleased keys
    if active_keys:
        print(f"\nUnreleased keys at end: {len(active_keys)}")
        for key, press_time in active_keys.items():
            print(f"  - {key}: pressed at {press_time}")
    
    # Issue breakdown
    issue_types = defaultdict(int)
    for issue in issues:
        issue_types[issue['type']] += 1
    
    print("\nIssue breakdown:")
    for issue_type, count in issue_types.items():
        print(f"  - {issue_type}: {count}")
    
    # Show sample issues
    if issues:
        print("\nSample issues (first 10):")
        for issue in issues[:10]:
            if issue['type'] == 'double_press':
                print(f"  - Double press: '{issue['key']}' at index {issue['index']}")
                print(f"    First press: {issue['first_press']}, Second press: {issue['second_press']}")
            elif issue['type'] == 'orphan_release':
                print(f"  - Orphan release: '{issue['key']}' at index {issue['index']}, time: {issue['time']}")
            elif issue['type'] == 'negative_hold_time':
                print(f"  - Negative hold time: '{issue['key']}' at index {issue['index']}")
                print(f"    Press: {issue['press_time']}, Release: {issue['release_time']}, Hold: {issue['hold_time']}")
    
    # Key-specific statistics
    print("\nKeys with issues:")
    for key, stats in sorted(key_stats.items(), key=lambda x: x[1]['issues'], reverse=True):
        if stats['issues'] > 0:
            print(f"  - {key}: {stats['presses']} presses, {stats['releases']} releases, {stats['issues']} issues")
    
    return issues, key_stats

def find_consecutive_issues(events):
    """Find cases where consecutive keys show the problematic pattern."""
    print("\nLooking for consecutive key timing issues...")
    
    skip_keys = {'Key.shift', 'Key.ctrl', 'Key.alt', 'Key.cmd', 'Key.caps_lock'}
    consecutive_issues = []
    
    # Look for pattern: P(key1), P(key2), R(key1), R(key2)
    for i in range(len(events) - 3):
        e1, e2, e3, e4 = events[i:i+4]
        
        # Skip if any are modifier keys
        if any(e['key'] in skip_keys for e in [e1, e2, e3, e4]):
            continue
            
        # Check for the problematic pattern
        if (e1['type'] == 'P' and e2['type'] == 'P' and 
            e3['type'] == 'R' and e4['type'] == 'R' and
            e1['key'] == e3['key'] and e2['key'] == e4['key']):
            
            time_diff = e2['time'] - e1['time']
            consecutive_issues.append({
                'keys': f"{e1['key']} -> {e2['key']}",
                'time_between_presses': time_diff,
                'index': i
            })
    
    if consecutive_issues:
        print(f"Found {len(consecutive_issues)} consecutive key timing patterns:")
        for issue in consecutive_issues[:10]:  # Show first 10
            print(f"  - {issue['keys']}: {issue['time_between_presses']}ms between presses (index {issue['index']})")
    
    return consecutive_issues

def compare_implementations(js_events, wasm_events):
    """Compare JavaScript and WASM implementations."""
    print(f"\n{'='*60}")
    print("Comparison between JavaScript and WASM")
    print(f"{'='*60}")
    
    js_issues, js_stats = analyze_key_order(js_events, "JavaScript")
    wasm_issues, wasm_stats = analyze_key_order(wasm_events, "WASM")
    
    # Find consecutive issues in both
    print(f"\n{'='*60}")
    print("Consecutive Key Analysis")
    print(f"{'='*60}")
    
    print("\nJavaScript:")
    js_consecutive = find_consecutive_issues(js_events)
    
    print("\nWASM:")
    wasm_consecutive = find_consecutive_issues(wasm_events)
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("Summary Comparison")
    print(f"{'='*60}")
    print(f"JavaScript: {len(js_issues)} total issues, {len(js_consecutive)} consecutive key patterns")
    print(f"WASM: {len(wasm_issues)} total issues, {len(wasm_consecutive)} consecutive key patterns")
    
    # Timing precision analysis
    print("\nTiming precision analysis:")
    js_times = [e['time'] for e in js_events]
    wasm_times = [e['time'] for e in wasm_events]
    
    if len(js_times) > 1:
        js_deltas = [js_times[i+1] - js_times[i] for i in range(len(js_times)-1)]
        print(f"JavaScript - Min delta: {min(js_deltas)}ms, Max delta: {max(js_deltas)}ms")
    
    if len(wasm_times) > 1:
        wasm_deltas = [wasm_times[i+1] - wasm_times[i] for i in range(len(wasm_times)-1)]
        print(f"WASM - Min delta: {min(wasm_deltas)}ms, Max delta: {max(wasm_deltas)}ms")

def main():
    if len(sys.argv) != 3:
        print("Usage: python analyze_keystrokes.py <javascript_csv> <wasm_csv>")
        sys.exit(1)
    
    js_file = sys.argv[1]
    wasm_file = sys.argv[2]
    
    try:
        js_events = load_csv(js_file)
        wasm_events = load_csv(wasm_file)
        
        compare_implementations(js_events, wasm_events)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

