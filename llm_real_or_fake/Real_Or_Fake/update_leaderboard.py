import argparse
import csv
import os
from datetime import datetime

parser = argparse.ArgumentParser(description="Update leaderboard")
parser.add_argument('--team', required=True)
parser.add_argument('--run', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--type', required=True)
parser.add_argument('--f1', type=float, required=True)
parser.add_argument('--accuracy', type=float, required=True)
args = parser.parse_args()

LEADERBOARD_FILE = 'docs/leaderboard.csv'
fieldnames = ['rank','team','run_id','model','model_type','f1_score','accuracy','submission_date']

# Load existing leaderboard
if os.path.exists(LEADERBOARD_FILE):
    with open(LEADERBOARD_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        leaderboard = list(reader)
else:
    leaderboard = []

# Convert f1_score and accuracy to float for sorting later
for row in leaderboard:
    row['f1_score'] = float(row['f1_score'])
    row['accuracy'] = float(row['accuracy'])

# Update or append submission
updated = False
for row in leaderboard:
    if row['team'] == args.team :
        row['model'] = args.model
        row['model_type'] = args.type
        row['f1_score'] = args.f1
        row['accuracy'] = args.accuracy
        row['submission_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated = True
        break

if not updated:
    leaderboard.append({
        'rank': '',  # will compute after sorting
        'team': args.team,
        'run_id': args.run,
        'model': args.model,
        'model_type': args.type,
        'f1_score': args.f1,
        'accuracy': args.accuracy,
        'submission_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# Sort leaderboard by f1_score descending, then accuracy descending
leaderboard.sort(key=lambda x: (x['f1_score'], x['accuracy']), reverse=True)

# Assign ranks
for i, row in enumerate(leaderboard, start=1):
    row['rank'] = i

# Save leaderboard
with open(LEADERBOARD_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(leaderboard)