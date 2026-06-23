from src.infrastructure.in_memory import (
    InMemoryPaperRepository,
    InMemoryPatentRepository,
    InMemoryProjectRepository,
    InMemoryResearcherRepository,
    InMemoryTechnologyRepository,
    InMemoryBudgetRepository,
)

r = InMemoryPaperRepository().search_papers("neuromorphic", 3)
assert len(r) > 0, "논문 검색 결과 없음"
print("논문:", r[0]["title"][:50])

r = InMemoryPatentRepository().search_patents("AI", "KR")
assert len(r) > 0, "특허 검색 결과 없음"
print("특허:", r[0]["title"][:50])

r = InMemoryProjectRepository().search_projects("PIM", 2022)
assert len(r) > 0, "국가과제 검색 결과 없음"
print("과제:", r[0]["title"][:50])

r = InMemoryResearcherRepository().recommend_researchers("PIM neuromorphic", 3)
assert len(r) > 0, "연구자 추천 결과 없음"
print("연구자:", [x["name"] for x in r])

r = InMemoryTechnologyRepository().recommend_technologies("메모리", 3)
assert len(r) > 0, "기술 추천 결과 없음"
print("기술:", r[0]["name"])

r = InMemoryBudgetRepository().analyze_budget("AI 반도체", 3)
assert r["cagr_percent"] > 0, "예산 CAGR 없음"
print("예산 CAGR:", r["cagr_percent"])

print("ALL OK")
