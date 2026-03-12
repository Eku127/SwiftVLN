"""
从 SatNav ver_260306 中抽取多样化 smoke test 子集。

策略：
  - 覆盖全部 9 个城市
  - 每个城市按 episode 长度分 3 档（短/中/长）各抽若干条
  - 使用固定随机种子，确保可复现

输出：
  baseline/smoke_test_data/annotations.json   （共 ~90 条）
"""

import json
import random
from collections import defaultdict
from pathlib import Path

SRC = Path("/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260306/trajectory_data/annotations.json")
OUT = Path(__file__).parent / "annotations.json"

SEED = 42
PER_CITY_SHORT  = 3   # steps < 30
PER_CITY_MEDIUM = 5   # 30 <= steps < 80
PER_CITY_LONG   = 3   # steps >= 80


def get_city(ep: dict) -> str:
    # video: "images/Geneva-1_satnav_000000" → "Geneva-1"
    return ep["video"].split("images/")[1].split("_satnav")[0]


def main():
    rng = random.Random(SEED)

    with open(SRC) as f:
        data = json.load(f)

    # 按城市 + 长度分档
    buckets: dict[str, dict[str, list]] = defaultdict(lambda: {"short": [], "medium": [], "long": []})
    for ep in data:
        city = get_city(ep)
        n = ep["steps"]
        if n < 30:
            buckets[city]["short"].append(ep)
        elif n < 80:
            buckets[city]["medium"].append(ep)
        else:
            buckets[city]["long"].append(ep)

    sampled = []
    for city in sorted(buckets):
        b = buckets[city]
        for key, quota in [("short", PER_CITY_SHORT), ("medium", PER_CITY_MEDIUM), ("long", PER_CITY_LONG)]:
            pool = b[key]
            rng.shuffle(pool)
            sampled.extend(pool[:quota])
        print(f"  {city}: short={len(b['short'])}, medium={len(b['medium'])}, long={len(b['long'])}")

    # 重新分配连续 id（保留原始字段，追加 smoke_source_id 方便回查）
    result = []
    for new_id, ep in enumerate(sampled):
        new_ep = dict(ep)
        new_ep["smoke_source_id"] = ep["id"]
        new_ep["id"] = new_id
        result.append(new_ep)

    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n✅ Saved {len(result)} episodes → {OUT}")
    # 统计摘要
    cities_out = defaultdict(int)
    for ep in result:
        cities_out[get_city(ep)] += 1
    print("\nDistribution:")
    for c, cnt in sorted(cities_out.items()):
        print(f"  {c}: {cnt}")
    lens = [ep["steps"] for ep in result]
    print(f"\nSteps: min={min(lens)}, max={max(lens)}, median={sorted(lens)[len(lens)//2]}")


if __name__ == "__main__":
    main()
