"""GPA goal planner math — pure functions, no LLM needed."""

GRADE_POINTS = {"A": 4.0, "A-": 3.7, "B+": 3.3, "B": 3.0, "B-": 2.7,
                "C+": 2.3, "C": 2.0, "C-": 1.7, "D+": 1.3, "D": 1.0, "F": 0.0}


def required_term_gpa(current_gpa: float, completed_credits: float,
                      target_gpa: float, planned_credits: float) -> float:
    """Average GPA needed across planned credits to reach target cumulative GPA."""
    if planned_credits <= 0:
        raise ValueError("Planned credits must be positive.")
    total_points_needed = target_gpa * (completed_credits + planned_credits)
    current_points = current_gpa * completed_credits
    return (total_points_needed - current_points) / planned_credits


def projected_gpa(current_gpa: float, completed_credits: float,
                  planned: list[dict]) -> float:
    """planned: [{'credits': c, 'grade': 'A'}, ...] -> resulting cumulative GPA."""
    pts = current_gpa * completed_credits
    creds = completed_credits
    for p in planned:
        pts += GRADE_POINTS[p["grade"]] * p["credits"]
        creds += p["credits"]
    return pts / creds if creds else 0.0


def min_grades_for_target(current_gpa: float, completed_credits: float,
                          target_gpa: float, planned: list[dict]) -> list[dict]:
    """For each planned course, the minimum letter grade needed if all other
    planned courses earn an A. Returns courses annotated with 'min_grade'."""
    out = []
    for i, p in enumerate(planned):
        others_pts = sum(GRADE_POINTS["A"] * q["credits"] for j, q in enumerate(planned) if j != i)
        others_creds = sum(q["credits"] for j, q in enumerate(planned) if j != i)
        need = target_gpa * (completed_credits + sum(q["credits"] for q in planned)) \
            - current_gpa * completed_credits - others_pts
        need_per_credit = need / p["credits"] if p["credits"] else 99
        best = "impossible"
        for g, pts in sorted(GRADE_POINTS.items(), key=lambda kv: -kv[1]):
            if pts >= need_per_credit - 1e-9:
                best = g
        out.append({**p, "min_grade": best})
    return out


def feasibility(required: float) -> str:
    if required > 4.0:
        return "Not possible — even straight A's won't reach it. Consider a longer timeline or more credits."
    if required > 3.7:
        return "Very tough — needs near-perfect grades. Build a backup plan."
    if required > 3.3:
        return "Challenging but doable with strong study habits."
    if required > 3.0:
        return "Very achievable — stay consistent."
    return "Comfortable — you're in good shape."
