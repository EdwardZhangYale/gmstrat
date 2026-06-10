import gzip
import json
from collections import Counter
import matplotlib.pyplot as plt

# Code courtesy of Claude Sonnet 4.6

fname = 'grid10x10-2'


def parse_districting(districting_list):
    """
    Convert the districting list-of-dicts format into a
    dict mapping district_id -> frozenset of precinct_ids.
    """
    district_map = {}
    for entry in districting_list:
        for precinct_id, district_id in entry.items():
            district_map.setdefault(district_id, []).append(precinct_id)
    return {d: frozenset(precincts) for d, precincts in district_map.items()}


def count_districts_with_scores(jsonl_gz_path):
    district_counter = Counter()
    district_iso_scores = {}  # frozenset -> isoperimetric score

    with gzip.open(jsonl_gz_path, "rt") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line or line_idx < 3:
                continue

            plan = json.loads(line)

            # The scores list is ordered by district_id (1, 2, 3, ...)
            iso_scores = plan["data"]["get_isoperimetric_scores"]

            # Parse districting and pair each district with its score
            districts = parse_districting(plan["districting"])

            for district_id, district_frozenset in districts.items():
                district_counter[district_frozenset] += 1

                # Store score if not seen before (score is a property of the
                # district geometry, so it's the same every time it appears)
                if district_frozenset not in district_iso_scores:
                    # district_id is 1-indexed, scores list is 0-indexed
                    district_iso_scores[district_frozenset] = iso_scores[district_id - 1]

    return district_counter, district_iso_scores


# --- Run ---
counter, iso_scores = count_districts_with_scores("local/output/" + fname + "/atlas.jsonl.gz")

# Most common districts
# for district, count in counter.most_common(10):
#     print(f"Count {count}: {len(district)} precincts, e.g. {sorted(district)[:5]}...")

output = [
    {
        "count": count,
        "isoperimetric_score": iso_scores[district_frozenset],
        "precincts": sorted(district_frozenset)
    }
    for district_frozenset, count in counter.most_common()
]

with open('local/output/' + fname + '/district_counts.json', "w") as f:
    json.dump(output, f, indent=2)

counts = [entry["count"] for entry in output]
scores = [entry["isoperimetric_score"] for entry in output]

plt.figure(figsize=(8, 5))
plt.scatter(scores, counts, alpha=0.5, s=10)
plt.yscale("log")
plt.xlabel("Isoperimetric Score")
plt.ylabel("Number of Occurrences (log scale)")
plt.title("District Frequency vs. Isoperimetric Score")
plt.tight_layout()
# plt.savefig("iso_vs_count.png", dpi=150)
plt.show()