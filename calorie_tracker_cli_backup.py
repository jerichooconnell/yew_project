#!/usr/bin/env python3
"""
Calorie Tracker - 6-Week Weight Loss Goal (900 kcal/day deficit)
A CLI app to track daily calories and progress toward a 6-week fat-loss goal.
Data is saved to calorie_tracker_data.json in the current directory.
"""

import json
import os
from datetime import date, datetime

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calorie_tracker_data.json")

# ── Constants ─────────────────────────────────────────────────────────────────

KCAL_PER_LB          = 3500   # dietary estimate: 1 lb of body fat ≈ 3,500 kcal
GOAL_WEEKS           = 6
GOAL_DEFICIT_PER_DAY = 900
# Fat burned at target deficit over the full goal period
GOAL_LBS = round(GOAL_DEFICIT_PER_DAY * 7 * GOAL_WEEKS / KCAL_PER_LB, 1)  # ≈ 10.8 lbs

# Daily fat-burned comparisons ordered by lbs (ascending).
# Each entry: (threshold_lbs, singular_label, plural_label)
DAILY_FAT_ITEMS = [
    (0.014,  "a sugar packet",           "sugar packets"),        # ~50 kcal each
    (0.057,  "a AA battery",             "AA batteries"),         # 0.057 lbs
    (0.101,  "a golf ball",              "golf balls"),           # 1.62 oz
    (0.25,   "a stick of butter",        "sticks of butter"),     # 4 oz
    (0.5,    "a baseball",               "baseballs"),            # ~0.32 lbs (close enough)
    (1.0,    "a can of soup",            "cans of soup"),
    (2.0,    "a large chicken breast",   "large chicken breasts"),
    (3.0,    "a hardcover book",         "hardcover books"),
    (5.0,    "a bag of flour",           "bags of flour"),
]

# Milestone messages: key = lbs burned (int), value = (headline, comparison object)
MILESTONES = {
    1:  ("*** 1 LB OF FAT BURNED! ***",
         "That's like carrying one less can of soup around all day."),
    2:  ("*** 2 LBS BURNED! Keep it up! ***",
         "Imagine two sticks of butter melted right off your body."),
    3:  ("*** 3 LBS GONE! Solid progress! ***",
         "You've shed the weight of a hardcover encyclopedia from pure effort."),
    4:  ("*** 4 LBS DOWN! Crushing it! ***",
         "That's the weight of a full laptop you've burned off. Incredible."),
    5:  ("*** 5 LBS BURNED! Almost halfway! ***",
         "A full 5 lb bag of flour — gone. You're unstoppable."),
    6:  ("*** 6 LBS! Over halfway there! ***",
         "That's a full half-gallon of milk melted off your frame."),
    7:  ("*** 7 LBS! Strong finish forming! ***",
         "Imagine 7 sticks of butter — that's the fat you've torched."),
    8:  ("*** 8 LBS DOWN! Almost there! ***",
         "A full 8-lb bowling ball of fat, gone for good."),
    9:  ("*** 9 LBS! One more to go! ***",
         "That's heavier than a newborn baby you've shed!"),
   10:  ("*** 10 LBS BURNED! Incredible! ***",
         "10 lbs — that's a large house cat's worth of fat, vanished."),
   11:  ("*** 6-WEEK GOAL CRUSHED! ***",
         "Over 10 lbs of fat burned through pure calorie discipline. You did it!"),
}

# ── Persistence ──────────────────────────────────────────────────────────────

def _migrate_profile(profile: dict) -> dict:
    """Add fields introduced after v1 without overwriting existing values."""
    profile.setdefault("start_weight", profile["weight"])
    profile.setdefault("milestones_shown", [])
    # v3: store the goal lbs so progress % works even if GOAL_LBS constant changes
    if "goal_lbs" not in profile:
        profile["goal_lbs"] = round(
            profile["start_weight"] - profile.get("goal_weight", profile["start_weight"] - 5), 1
        )
    # v3: store the per-day deficit target used when the profile was created
    if "deficit_target" not in profile:
        # infer from saved tdee and daily_budget; fall back to old default
        tdee   = profile.get("tdee", 0)
        budget = profile.get("daily_budget", 0)
        profile["deficit_target"] = (tdee - budget) if tdee and budget else 500
    return profile


def load_data() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
        # back-compat: ensure top-level keys exist
        data.setdefault("log", {})
        data.setdefault("weight_log", {})
        if data.get("profile"):
            data["profile"] = _migrate_profile(data["profile"])
        return data
    return {"profile": None, "log": {}, "weight_log": {}}


def save_data(data: dict) -> None:
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ── Calorie-deficit fat calculations ─────────────────────────────────────────

def calc_total_deficit_kcal(data: dict) -> float:
    """Sum of (budget − consumed) across all logged days. Negative = surplus."""
    budget = data["profile"]["daily_budget"]
    total = 0.0
    for entries in data["log"].values():
        consumed = sum(e["calories"] for e in entries)
        total += budget - consumed
    return total


def deficit_to_lbs(kcal: float) -> float:
    """Convert kcal deficit to approximate lbs of fat."""
    return kcal / KCAL_PER_LB


def fat_item_label(lbs: float) -> str:
    """Return a human-readable comparison for a given number of fat lbs."""
    if lbs <= 0:
        return "nothing yet — stay in a deficit!"
    for threshold, singular, plural in DAILY_FAT_ITEMS:
        if lbs < threshold * 1.5:
            count = lbs / threshold
            if count < 1.8:
                return f"about {singular}"
            return f"about {count:.1f} {plural}"
    # above all thresholds
    bags = lbs / 5.0
    return f"about {bags:.1f} bags of flour"


def day_deficit_lbs(data: dict, day_key: str) -> float:
    """Calorie-deficit-based fat burned on a specific day (lbs)."""
    budget   = data["profile"]["daily_budget"]
    entries  = data["log"].get(day_key, [])
    consumed = sum(e["calories"] for e in entries)
    return deficit_to_lbs(max(0, budget - consumed))


def check_milestones(data: dict) -> None:
    """Print congrats messages for any newly crossed lb milestones."""
    p = data["profile"]
    shown = p.setdefault("milestones_shown", [])
    total_deficit = calc_total_deficit_kcal(data)
    lbs_burned = deficit_to_lbs(total_deficit)

    for lb_mark, (headline, comparison) in MILESTONES.items():
        if lbs_burned >= lb_mark and lb_mark not in shown:
            print()
            print("  " + "★" * 48)
            print(f"  {headline}")
            print(f"  {comparison}")
            print("  " + "★" * 48)
            shown.append(lb_mark)

    save_data(data)

# ── Calculations ──────────────────────────────────────────────────────────────

ACTIVITY_LEVELS = {
    "1": ("Sedentary",        "Little or no exercise",                    1.2),
    "2": ("Lightly active",   "Light exercise 1-3 days/week",             1.375),
    "3": ("Moderately active","Moderate exercise 3-5 days/week",          1.55),
    "4": ("Very active",      "Hard exercise 6-7 days/week",              1.725),
    "5": ("Extra active",     "Very hard exercise or physical job",       1.9),
}

def calc_bmr(weight_lbs: float, height_in: float, age: int, sex: str) -> float:
    """Mifflin-St Jeor BMR (metric internally, convert inputs)."""
    weight_kg = weight_lbs * 0.453592
    height_cm = height_in * 2.54
    if sex.lower() == "m":
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161


def calc_tdee(bmr: float, activity_multiplier: float) -> float:
    return bmr * activity_multiplier


def calc_daily_budget(tdee: float, deficit: int = GOAL_DEFICIT_PER_DAY) -> int:
    """Return calorie budget. Default deficit targets GOAL_DEFICIT_PER_DAY kcal/day."""
    return max(1200, round(tdee - deficit))

# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_profile(data: dict) -> None:
    print("\n=== Profile Setup ===")
    name       = input("Your name: ").strip() or "User"
    weight     = float(input("Current weight (lbs): "))
    height_ft  = int(input("Height — feet: "))
    height_in  = int(input("            — inches: "))
    total_in   = height_ft * 12 + height_in
    age        = int(input("Age: "))
    sex        = ""
    while sex not in ("m", "f"):
        sex = input("Sex (m/f): ").strip().lower()

    print("\nActivity level:")
    for k, (label, desc, _) in ACTIVITY_LEVELS.items():
        print(f"  {k}. {label} – {desc}")
    act_key = ""
    while act_key not in ACTIVITY_LEVELS:
        act_key = input("Choose 1-5: ").strip()

    act_label, _, act_mult = ACTIVITY_LEVELS[act_key]
    bmr    = calc_bmr(weight, total_in, age, sex)
    tdee   = calc_tdee(bmr, act_mult)
    budget = calc_daily_budget(tdee)  # uses GOAL_DEFICIT_PER_DAY default
    goal_weight = weight - GOAL_LBS

    # Preserve milestones_shown and start_weight if profile already existed
    prev = data.get("profile") or {}
    data["profile"] = {
        "name":             name,
        "weight":           weight,
        "goal_weight":      round(goal_weight, 1),
        "goal_lbs":         GOAL_LBS,
        "deficit_target":   GOAL_DEFICIT_PER_DAY,
        "height_in":        total_in,
        "age":              age,
        "sex":              sex,
        "activity":         act_label,
        "act_mult":         act_mult,
        "bmr":              round(bmr),
        "tdee":             round(tdee),
        "daily_budget":     budget,
        "created":          prev.get("created", str(date.today())),
        "start_weight":     prev.get("start_weight", weight),
        "milestones_shown": prev.get("milestones_shown", []),
    }
    save_data(data)

    print(f"\n  BMR              : {round(bmr)} kcal")
    print(f"  TDEE             : {round(tdee)} kcal")
    print(f"  Daily budget     : {budget} kcal  ({GOAL_DEFICIT_PER_DAY} kcal deficit/day)")
    print(f"  Goal weight      : {goal_weight:.1f} lbs  (current {weight:.1f} lbs − {GOAL_LBS} lbs)")
    print(f"  Estimated time   : {GOAL_WEEKS} weeks")

# ── Today's log ───────────────────────────────────────────────────────────────

def today_key() -> str:
    return str(date.today())


def get_today_log(data: dict) -> list:
    return data["log"].setdefault(today_key(), [])


def log_food(data: dict) -> None:
    name     = input("Food / meal name: ").strip()
    if not name:
        print("Cancelled.")
        return
    calories = input("Calories: ").strip()
    if not calories.lstrip("-").isdigit():
        print("Invalid calories, cancelled.")
        return
    calories = int(calories)
    if calories < 0:
        print("Calories can't be negative, cancelled.")
        return

    entry = {
        "time":     datetime.now().strftime("%H:%M"),
        "name":     name,
        "calories": calories,
    }
    get_today_log(data).append(entry)
    save_data(data)
    budget    = data["profile"]["daily_budget"]
    consumed  = sum(e["calories"] for e in get_today_log(data))
    remaining = budget - consumed
    print(f"  Logged {calories} kcal. Today: {consumed}/{budget} kcal  ({remaining:+d} remaining)")

    # Show today's fat-burned comparison
    today_lbs = day_deficit_lbs(data, today_key())
    if today_lbs > 0:
        item = fat_item_label(today_lbs)
        print(f"  Today's deficit burns ~{today_lbs:.3f} lbs of fat  ({item})")

    # Check and announce any newly crossed milestones
    check_milestones(data)


def show_today(data: dict) -> None:
    p = data["profile"]
    entries = get_today_log(data)
    consumed = sum(e["calories"] for e in entries)
    budget = p["daily_budget"]
    remaining = budget - consumed

    print(f"\n=== {today_key()} — Daily Log ===")
    if not entries:
        print("  No entries yet.")
    else:
        for e in entries:
            print(f"  {e['time']}  {e['name']:<30} {e['calories']:>5} kcal")
    print(f"\n  Budget   : {budget} kcal")
    print(f"  Consumed : {consumed} kcal")
    if remaining >= 0:
        print(f"  Remaining: {remaining} kcal")
    else:
        print(f"  OVER by  : {-remaining} kcal  ⚠")

    # Fat burned so far today
    today_lbs = day_deficit_lbs(data, today_key())
    if today_lbs > 0:
        item = fat_item_label(today_lbs)
        print(f"\n  Fat burned today : ~{today_lbs:.3f} lbs  ({item})")
    else:
        print(f"\n  Fat burned today : 0 lbs (still over budget)")


def remove_entry(data: dict) -> None:
    entries = get_today_log(data)
    if not entries:
        print("No entries to remove.")
        return
    for i, e in enumerate(entries, 1):
        print(f"  {i}. {e['time']}  {e['name']}  ({e['calories']} kcal)")
    choice = input("Remove entry #: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(entries)):
        print("Invalid selection.")
        return
    removed = entries.pop(int(choice) - 1)
    save_data(data)
    print(f"  Removed: {removed['name']} ({removed['calories']} kcal)")

# ── Weight log ────────────────────────────────────────────────────────────────

def log_weight(data: dict) -> None:
    p = data["profile"]
    w = input(f"Today's weight (lbs) [current: {p['weight']}]: ").strip()
    if not w:
        return
    try:
        w = float(w)
    except ValueError:
        print("Invalid weight.")
        return
    if w <= 0 or w > 1000:
        print("Implausible weight value, cancelled.")
        return

    data["weight_log"][today_key()] = w
    p["weight"] = w
    save_data(data)

    goal        = p["goal_weight"]
    goal_lbs    = p.get("goal_lbs", GOAL_LBS)
    start       = p.get("start_weight", goal + goal_lbs)
    scale_lost  = start - w
    remaining   = w - goal

    # Calorie-deficit based fat loss
    deficit_kcal = calc_total_deficit_kcal(data)
    deficit_lbs  = deficit_to_lbs(deficit_kcal)
    item         = fat_item_label(deficit_lbs)

    print(f"  Logged {w} lbs.")
    print(f"  Scale weight lost : {scale_lost:.1f} lbs  (from {start:.1f} lbs start)")
    print(f"  Still to go       : {remaining:.1f} lbs toward {goal:.1f} lb goal  (−{goal_lbs} lb target)")
    print(f"  Calorie-deficit   : ~{deficit_lbs:.2f} lbs of fat burned  ({item})")

    check_milestones(data)


def show_progress(data: dict) -> None:
    p    = data["profile"]
    wlog = data["weight_log"]
    start_weight = p.get("start_weight", p["weight"])
    goal_lbs     = p.get("goal_lbs", GOAL_LBS)
    deficit_tgt  = p.get("deficit_target", GOAL_DEFICIT_PER_DAY)

    # ── Calorie-deficit fat loss ───────────────────────────────────────────
    deficit_kcal = calc_total_deficit_kcal(data)
    deficit_lbs  = max(0.0, deficit_to_lbs(deficit_kcal))
    pct_deficit  = min(100.0, deficit_lbs / goal_lbs * 100)
    bar_d        = int(pct_deficit / 5)
    bar_deficit  = "█" * bar_d + "░" * (20 - bar_d)

    # ── Scale-weight loss ─────────────────────────────────────────────────
    scale_lost   = start_weight - p["weight"]
    pct_scale    = max(0, min(100, scale_lost / goal_lbs * 100))
    bar_s        = int(pct_scale / 5)
    bar_scale    = "█" * bar_s + "░" * (20 - bar_s)

    print(f"\n=== Weight Progress ===")
    print(f"  Plan   : {deficit_tgt} kcal/day deficit over {GOAL_WEEKS} weeks")
    print(f"  Start  : {start_weight:.1f} lbs  →  Goal : {p['goal_weight']:.1f} lbs  (−{goal_lbs} lbs)")
    print(f"  Current scale weight : {p['weight']:.1f} lbs")
    print()
    print(f"  Calorie deficit (fat burned):")
    print(f"    [{bar_deficit}] {pct_deficit:.0f}%  (~{deficit_lbs:.2f}/{goal_lbs} lbs)")
    print(f"    That's like  : {fat_item_label(deficit_lbs)}")
    print(f"    Total deficit: {deficit_kcal:,.0f} kcal")
    print()
    print(f"  Scale weight lost:")
    print(f"    [{bar_scale}] {pct_scale:.0f}%  ({scale_lost:.1f}/{goal_lbs} lbs)")

    # ── Milestone comparisons ─────────────────────────────────────────────
    relevant = {k: v for k, v in MILESTONES.items() if k <= goal_lbs + 1}
    print()
    print("  ── Milestone reference ──────────────────────────────────")
    for lb_mark, (headline, comparison) in relevant.items():
        reached = "✓" if deficit_lbs >= lb_mark else "○"
        print(f"  {reached} {lb_mark:>2} lb : {comparison}")

    remaining_scale = p["weight"] - p["goal_weight"]
    print()
    if deficit_lbs >= goal_lbs or remaining_scale <= 0:
        print("  *** GOAL REACHED! Congratulations! ***")
    else:
        lbs_left = goal_lbs - deficit_lbs
        weeks_left = lbs_left * KCAL_PER_LB / (deficit_tgt * 7)
        print(f"  Remaining : ~{lbs_left:.1f} lbs  (~{weeks_left:.1f} week(s) at {deficit_tgt} kcal/day pace)")

    if wlog:
        print("\n  ── Weigh-in history ─────────────────────────────────────")
        print(f"  {'Date':<12} {'Weight':>8}")
        for d in sorted(wlog):
            print(f"  {d}   {wlog[d]:.1f} lbs")


def show_weekly_summary(data: dict) -> None:
    from datetime import timedelta
    p   = data["profile"]
    log = data["log"]
    budget = p["daily_budget"]

    print("\n=== Last 7 Days ===")
    print(f"  {'Date':<12} {'Consumed':>9} {'Budget':>7} {'Diff':>6}")
    print("  " + "-" * 38)
    today = date.today()
    for i in range(6, -1, -1):
        d = str(today - timedelta(days=i))
        entries = log.get(d, [])
        consumed = sum(e["calories"] for e in entries)
        diff = consumed - budget
        marker = " ✓" if diff <= 0 else " ✗"
        if consumed == 0 and d not in log:
            print(f"  {d:<12} {'—':>9}")
        else:
            print(f"  {d:<12} {consumed:>9} {budget:>7} {diff:>+6}{marker}")

# ── Main menu ─────────────────────────────────────────────────────────────────

def print_header(data: dict) -> None:
    p = data.get("profile")
    print("\n" + "=" * 50)
    print(f"  Calorie Tracker — {GOAL_WEEKS}-Week Goal ({GOAL_DEFICIT_PER_DAY} kcal/day deficit)")
    print("=" * 50)
    if p:
        budget   = p["daily_budget"]
        consumed = sum(e["calories"] for e in get_today_log(data))
        remaining = budget - consumed
        goal     = p["goal_weight"]
        current  = p["weight"]
        print(f"  {p['name']} | Today: {consumed}/{budget} kcal ({remaining:+d}) | {current:.1f}/{goal:.1f} lbs")


def main() -> None:
    data = load_data()

    if data["profile"] is None:
        print("Welcome to Calorie Tracker!")
        print("Let's set up your profile to calculate your daily calorie budget.")
        setup_profile(data)

    while True:
        print_header(data)
        print()
        print("  1. Log food / meal")
        print("  2. View today's log")
        print("  3. Remove a food entry")
        print("  4. Log today's weight")
        print("  5. View weight progress")
        print("  6. Weekly calorie summary")
        print("  7. Edit profile / recalculate budget")
        print("  8. Quit")
        print()

        choice = input("  Choose an option: ").strip()

        if choice == "1":
            log_food(data)
        elif choice == "2":
            show_today(data)
        elif choice == "3":
            remove_entry(data)
        elif choice == "4":
            log_weight(data)
        elif choice == "5":
            show_progress(data)
        elif choice == "6":
            show_weekly_summary(data)
        elif choice == "7":
            setup_profile(data)
        elif choice == "8":
            print("Goodbye!")
            break
        else:
            print("  Invalid choice, try again.")


if __name__ == "__main__":
    main()
