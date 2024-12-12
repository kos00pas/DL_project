abstract 

        stop_keywords = ["Introduction", "Background", "Conclusion", "References", "Acknowledgments", "Keywords","Related Work","$"]
        abstract_pattern = rf"(?:^|\n)\s*(Abstract|Summary|Overview|Synopsis)\s*[:.\-]?\s*(.*?)(?=\n\s*({'|'.join(stop_keywords)}))"

instead 

         abstract_pattern = r"(?:^|\n)\s*(Abstract)\s*[:.\-]?\s*(.*?)(?=\n\s*(Introduction|Background|Conclusion|References|$))"


---
add in :  extract_sections_from_pdf

        problem_and_solution_pattern = (
                    r"(?:^|\n)\s*(Problem Statement|Challenges|Motivation|Existing Gaps|Contributions|Our Approach|Proposed Solution|Novelty|Innovative Approach)"
                    r"\s*[:.\-]?\s*(.*?)"
                    r"(?=\n\s*(Method|Results|Discussion|Conclusion|References|Acknowledgments|Appendix|$))"
                )

---

