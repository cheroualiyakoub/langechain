"""
Patent claim generation prompt engineering module.

This module contains prompt templates and instructions for generating
patent claims from technical descriptions using language models.
"""

# Patent claim generation prompt
PATENT_CLAIM_GENERATION_PROMPT = """
You are a patent claim generation assistant. Given a technical description of an invention, your task is to extract **all possible patent claims** that can be supported by the content.

Follow these steps strictly:

---

1. **Analyze the Description**:
   - Identify the **core invention or novel idea**.
   - Detect all **technical components**, **methods**, or **features** that contribute to functionality or novelty.

2. **Generate Patent Claims**:
   - Write the claims in **clear, formal language**.
   - Start with **at least one independent claim** (method or system).
   - Follow with **dependent claims** that add details, such as:
     - Specific materials used
     - Geometric shapes
     - Placement techniques
     - Functional enhancements
     - Environmental variations (e.g., under heat or torque)
     - Multi-mode tuning (e.g., shell, bending, torsion)
     - Manufacturing methods

3. **Structure**:
   - Number each claim clearly (e.g., Claim 1, Claim 2, ...)
   - Avoid repeating elements already claimed in parent claims—just refer to them (e.g., "The method of Claim 1, wherein...")

4. **Maximize Coverage**:
   - Cover all *possible uses* or *configurations* hinted at in the description.
   - Suggest broader **alternative embodiments**.
   - Add *claims around tuning*, *placement*, *absorption methods*, *mechanical integration*, and *material properties* if present.

---

Now, based on the following description, generate all possible patent claims:
{description}
"""

# Function to get the prompt with a description inserted
def get_patent_claim_prompt(description: str) -> str:
    """
    Get the patent claim generation prompt with the provided description inserted.
    
    Args:
        description: The technical description of the invention
        
    Returns:
        A formatted prompt ready to be sent to an LLM
    """
    return PATENT_CLAIM_GENERATION_PROMPT.format(description=description)


# Example usage of how to use this module:
if __name__ == "__main__":
    # Example technical description
    example_description = """
    A novel acoustic metamaterial comprising a periodic arrangement of resonating units.
    Each unit consists of a hollow cylinder with interior baffles positioned at specific
    intervals. The baffles are made of a composite material that includes a viscoelastic
    layer sandwiched between two rigid layers. This configuration enables selective
    frequency absorption across a wide bandwidth while maintaining structural integrity.
    """
    
    # Generate the prompt with the example description
    prompt = get_patent_claim_prompt(example_description)
    print(prompt)
    
    # In a real application, you would then send this prompt to your LLM
    # Example:
    # response = llm.generate(prompt)
    # print(response)