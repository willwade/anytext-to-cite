#!/usr/bin/env python3
"""
Test script for the split_references function in convert.py
"""

from convert import split_references

# Test cases
test_cases = [
    # Test case 1: Numbered references
    """
103. Community Health Workers in Low-and Middle-Income Countries ..., accessed on May 6, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC3682607/
104. Community Health Workers Can Provide Psychosocial Support to the People During COVID-19 and Beyond in Low- and Middle- Income Countries - Frontiers, accessed on May 6, 2025, https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2021.666753/full
    """,
    
    # Test case 2: URL references without numbering
    """
Public health surveillance through community health workers: a scoping review of evidence from 25 low-income and middle-income countries - CHW Central, accessed on May 6, 2025, https://chwcentral.org/resources/public-health-surveillance-through-community-health-workers-a-scoping-review-of-evidence-from-25-low-income-and-middle-income-countries/
Community Health Workers in Low-and Middle-Income Countries ..., accessed on May 6, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC3682607/
documents1.worldbank.org, accessed on May 6, 2025, https://documents1.worldbank.org/curated/en/099337012142213309/pdf/IDU02966dd930cfde041c308669055e3b8e316ad.pdf
    """,
    
    # Test case 3: Academic citations
    """
Prins, G. B., Nizeyimana, E., Ernstzen, D. V., & Louw, Q. A. (2024). Perspectives of patients with osteoarthritis for using digital technology in rehabilitation at a public community centre in the Cape Metropole area: A qualitative study. DIGITAL HEALTH, 10. https://doi.org/10.1177/20552076241282230

Mattke, S. (2011). Identifying and addressing obstacles to telecare uptake. International Journal of Integrated Care, 11(6). https://doi.org/10.5334/ijic.712

Bhattacharya, I., Ramachandran, A., Upadhyay, N., & Sharma, M. (2013). Assistive Computing Technology for Enabling Differently-Abled Population in India. International Journal of User-Driven Healthcare, 3(2), 33–43. https://doi.org/10.4018/ijudh.2013040104
    """
]

# Run tests
for i, test_case in enumerate(test_cases):
    print(f"\n=== Test Case {i+1} ===")
    references = split_references(test_case)
    print(f"Found {len(references)} references:")
    for j, ref in enumerate(references):
        print(f"{j+1}. {ref[:100]}..." if len(ref) > 100 else f"{j+1}. {ref}")
