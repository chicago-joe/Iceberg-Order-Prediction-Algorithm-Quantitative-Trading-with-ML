# instructions part 1

Using your internal project knowledge as the sole source of truth, generate an updated, advanced technical walkthrough in Markdown for parts.md. Do not use or reference any external resources. This final document must integrate detailed technical insights from all available internal documentation, including the Markdown (md) and HTML files, as well as the Python source code files. Follow these instructions exactly:

1. **Overall Task Breakdown with SequentialThinking MCP**
   
   - Utilize the sequentialThinking mcp and mcp-memory server to break the process into clear, manageable subtasks. Ensure that each subtask is processed independently while maintaining continuity of context between tasks.
   - The high-level subtasks should include:
     - Extraction and cross-referencing of relevant content from parts.md, additional MD and HTML files, and Python source code.
     - Extraction and integration of images (including neptune.ai model results) with context-specific annotations.
     - Detailed technical explanations for:
       - ECS container orchestration and tick data simulation from `sxm_market_similator.py`.
       - CPU parallelization using Ray, supported by direct code quotes.
       - Machine learning integration from `machine_learning_final_modified.py`, incorporating only evidence-based metrics.
     - Final consolidation of all the above components into a single cohesive Markdown file.

2. **Cross-Reference Verification & Aggregation of Existing Details**
   
   - Systematically cross-check parts.md with all other internal documentation (MD and HTML) that include technical details.
   - Extract and include only those technical explanations and examples that are directly supported by your project's source code or internal documentation.
   - In instances of conflicting or overlapping information, select the content that best aligns with the source code evidence.

3. **Image and Data Integration**
   
   - Incorporate all relevant images (e.g., model results from neptune.ai) into the updated parts.md.
   - For each image:
     - If its content is clearly supported by specific source code sections, add a dedicated explanatory segment that includes direct code quotes and numerical data.
     - If the image’s relevance is ambiguous and cannot be directly evidenced by the code, insert it with a concise annotation stating “I don't know.”
   - Do not generate or infer any performance metrics (e.g., fill rates, slippage) unless explicitly documented.

4. **Detailed Technical Explanations for Quantitative Trading Applications**
   
   - Write an extremely detailed walkthrough aimed at a graduate-level quantitative trading audience.
   - **ECS and Tick Data Simulation:**
     - Use `sxm_market_similator.py` to detail the orchestration of ECS containers for simulating historical tick data.
     - Explain task distribution, container coordination, and simulation mechanisms without fabricating data.
   - **CPU Parallelization via Ray:**
     - Explain how tasks are parallelized across CPUs using Ray, including direct code excerpts that illustrate this functionality.
   - **Machine Learning Integration:**
     - Reference `machine_learning_final_modified.py` to describe the integration of machine learning algorithms into the project.
     - Include operational details and any evidence-based performance metrics only if directly supported by internal documentation.
   - Seamlessly integrate any excellent technical details extracted from the MD and HTML files that are consistent with the project code.

5. **Token and Memory Optimization**
   
   - Employ all available mcp and token optimization strategies to ensure efficient token and memory management.
   - Use the mcp-memory server to preserve state and transition information between the defined subtasks, ensuring the final document is consistent and complete.
   - Aim for concise yet comprehensive explanations without redundancy.

6. **Final Output**
   
   - Generate a new, cohesive Markdown file that merges:
     - The original content of parts.md (which must remain intact),
     - Integrated images with corresponding technical annotations,
     - Detailed technical explanations based solely on internal evidence from the project documentation and source code.
   - The final document should be a comprehensive and logically organized walkthrough that covers simulation, parallel processing, machine learning integration, and their applications in quantitative trading.

Remember: Every detail (including any metrics, code quotes, or operational insights) must be directly verifiable from your internal project files. Do not introduce or infer any new data unless it is explicitly supported by your internal documentation and source code.

# instructions part 2

SYSTEM PROMPT (Role Definition):
You are Claude, a sophisticated language model with advanced reasoning and composition capabilities, specialized in generating technical documentation for a quantitative trading project. Your priorities are:

1. Accuracy: Only use information verifiably present in internal project files (Python code, MD, HTML documentation).
2. Clarity: Present the technical detail in a structured, methodical manner (e.g., suitable for a graduate-level trading course).
3. Thoroughness: Integrate any relevant images, code snippets, and data from neptune.ai if (and only if) they can be directly substantiated by the internal docs.
4. Efficiency: Use token and memory optimizations via sequentialThinking mcp and mcp-memory server to break the task into subtasks, preserve state, and avoid redundancy.

OVERALL OBJECTIVE:
Create a new Markdown file that merges:

- The existing `parts.md` content (unchanged, but possibly expanded).
- Aggregated technical insights from multiple Markdown and HTML files containing overlapping or additional details.
- Verified references from the Python source code (`sxm_market_similator.py`, `machine_learning_final_modified.py`, etc.).
- Relevant images (including neptune.ai results), each explained if directly verifiable or labeled “I don't know” if not.

MAIN TASKS:

1. **Prompt Engineering**  
   
   - Apply Anthropic’s **Prompt Templates** and **Prompt Improver** concepts to maintain consistency in style and formatting.  
   - Write in a clear, direct style, as per best practices for **Be Clear and Direct**.

2. **Use Examples (Multishot Prompting)**  
   
   - Reference or insert short, relevant code examples from `sxm_market_similator.py` or `machine_learning_final_modified.py` to demonstrate how ECS containers are orchestrated or how Ray parallelizes CPU tasks.  
   - Where performance metrics exist (e.g., from neptune.ai), showcase them **only** if directly evidenced. If unknown or not explicitly stated, respond with “I don't know.”

3. **Let Claude Think (Chain-of-Thought)**  
   
   - Employ a short chain-of-thought internally to logically step through each subtask, ensuring that no detail is missed and that you do not fabricate or infer data points that lack source code support.  
   - Offer extended thinking tips for complex steps in the commentary so a user can follow your logical process.

4. **Use XML Tags (If Needed)**  
   
   - If structuring or parsing content from the large MD/HTML files, embed it with XML-like tags to systematically capture sections or references. Only do this if it helps maintain clarity.  
   - Make sure final user-facing output is still in Markdown.

5. **Give Claude a Role (System Prompts)**  
   
   - Reinforce the system prompt role: you are a rigorous technical documenter.  

6. **Prefill Claude’s Response**  
   
   - You may start each subtask with short bullet points or placeholders (like “**Subtask 1**: Cross-referencing the code…”) to keep track of your flow, as suggested by Anthropic’s guidelines on prefill responses.

7. **Chain Complex Prompts**  
   
   - Break the overall prompt into sequential steps with **sequentialThinking mcp**. Each step can gather relevant info from your memory (mcp-memory server), then feed that into the next.  

8. **Long Context Tips**  
   
   - For thousands of lines of MD and HTML documentation, chunk them into smaller sections. Summarize each chunk’s key insights and confirm relevance to the source code.  
   - This mitigates token usage overload and helps ensure important details aren’t lost.

9. **Extended Thinking Tips**  
   
   - For especially intricate logic (e.g., explaining ECS concurrency or advanced ML techniques), provide a thorough breakdown.  
   - Then highlight how it applies to a quantitative trading environment: e.g., how parallelization can reduce latencies or how ML features might predict market microstructure anomalies.

10. **Extended Thinking**  
    
    - Where needed, use more advanced reasoning to unify the explanations of ECS, Ray, ML, and any performance measures.  
    - If certain metrics (fill rates, slippage) are not found, insert “I don’t know.”

11. **Multilingual Support** *(Optional)*  
    
    - If relevant, clarify if any text is required in another language—but by default, produce the output in English for a graduate-level audience.

12. **Tool Use (Function Calling)**  
    
    - If any function calls or special syntax is required (like reading chunked HTML, searching the code for references, etc.), you may structure your solution accordingly.  

13. **Prompt Caching**  
    
    - Keep a cache or running memory (mcp-memory) of previously processed chunks so each subsequent step doesn’t re-parse the entire data.  

14. **PDF Support** *(Optional)*  
    
    - If there are PDFs (not mentioned in the prompt, but for completeness), process them similarly by chunking. Not applicable if not needed.

15. **Citations**  
    
    - Provide citations or references to specific lines of the source code or documentation. For instance, “As per `sxm_market_similator.py` lines 123-145…”  

16. **Token Counting**  
    
    - Ensure the final answer doesn’t exceed the maximum tokens. Summarize or chunk data as necessary.  

17. **Batch Processing**  
    
    - If dealing with multiple large files, process them in discrete batches.  

18. **Embeddings** *(Optional)*  
    
    - For searching relevant segments of code or text, embeddings may be used behind the scenes. Not mandatory, but can be beneficial if available.

19. **Strict Evidence-Only Outputs**  
    
    - Do **not** fabricate any metrics, fill rates, slippage, or performance numbers. Provide only those that appear in the code or in the docs. If not stated, use “I don’t know.”  

20. **Final Output**  
    
    - Produce a **single Markdown file** capturing the expanded `parts.md` content with all newly integrated details, images, and references.  
    - Each image should be placed in context with an explanation or “I don’t know.”  
    - The entire document should feel like an advanced technical manual for quantitative traders, bridging ECS-based simulation, Ray parallelization, and machine learning modeling, supported by internal code references and data from neptune.ai **wherever it is explicitly available**.

# instructions part 3 - neptune

Using your internal project knowledge as the sole source of truth, generate an extended and advanced technical walkthrough in Markdown that includes detailed analysis and visualizations based on the model run metadata for two experiments: sxm-449 and sxm-762. Note that the complete model run metadata (including artifacts, tuning parameters, training details, and evaluation metrics) has been downloaded and provided as ZIP files. Do not attempt to retrieve additional data—use only the metadata available in these ZIP files.

1. **Metadata Integration and Analysis:**
   
   - Extract, read, and parse the provided ZIP files for sxm-449 and sxm-762.
   - For sxm-449, note that it contains Optuna integration details. Identify all relevant fields that document tuning iterations, hyperparameter choices, training loss, performance metrics, and model-specific parameters.
   - For sxm-762, similarly extract all available metadata fields.
   - Compare the metadata for both runs. Identify common fields and any differences, especially highlighting the additional details from the Optuna integration in sxm-449.

2. **Charting and Visualization Opportunities:**
   
   - Based on the metadata fields, determine which numeric or categorical values can be plotted. This includes iterations vs. training loss, hyperparameter tuning progress, feature importance trends, and any other model performance metrics.
   - Suggest the best chart types (e.g., line charts, scatter plots, bar charts) to visually compare these aspects.
   - Provide reasoning on how these visualizations can help interpret the impact of tuning and training differences on model performance. If any field or metric is not directly evidenced in the metadata, explicitly respond with “I don’t know.”

3. **Technical Walkthrough Integration:**
   
   - Incorporate the insights from the metadata analysis into an extended technical walkthrough. This walkthrough should be written in Markdown and integrate with the existing parts.md.
   - Include detailed explanations of:
     - How metadata was parsed and interpreted.
     - The steps taken to compare runs sxm-449 and sxm-762.
     - The integration of Optuna metrics in sxm-449 and their effect on tuning.
     - The design choices for charting—supported by examples from the metadata fields.
   - Use direct quotes or excerpts from the metadata files where applicable (only if they are verifiable by the provided data).

4. **Sequential Subtask Management:**
   
   - Use the sequentialThinking mcp and mcp-memory server to process this task in logical steps:
     - (a) Parsing and extraction of metadata from the ZIP files.
     - (b) Comparative analysis of the two runs.
     - (c) Identification of viable charts and plots.
     - (d) Merging the analysis into the final Markdown writeup.
   - Ensure that state is preserved between the subtasks to maintain context and consistency.

5. **Final Output:**
   
   - Generate a single cohesive Markdown file that includes:
     - The complete analysis of model run metadata.
     - Comparative charts and plot design suggestions.
     - Detailed, evidence-based explanations for each visualization, ensuring that every metric is directly supported by the provided metadata.  
   - The document should serve as an advanced technical guide for a quantitative trading audience, explaining how metadata is used to compare and evaluate model performance without introducing any fabricated values.

Remember:

- Do not invent any new metadata values—rely solely on what is present in the provided ZIP files.
- If any metadata field or detail is ambiguous or missing, clearly indicate “I don’t know.”
- Use clear, concise, and graduate-level technical language that bridges the gap between code documentation and real-world quantitative trading applications.

# instructions backup neptune

Using your internal project knowledge as the sole source of truth, and without referencing any external resources beyond the provided documents, generate an extended Markdown technical writeup that covers the following steps and details. Your target audience is advanced quantitative trading professionals, so your explanation must be thorough, evidence-based, and highly technical.

1. **Documentation Analysis for Neptune Integrations:**
   
   - Analyze and summarize the key insights from the following Neptune integration and metadata documentation URLs:
     - https://docs.neptune.ai/integrations/xgboost/
     - https://docs.neptune.ai/tutorials/
     - https://docs.neptune.ai/tutorials/basic_ml_run_tracking/#customize-column-appearance
     - https://docs.neptune.ai/usage/downloading_metadata/
     - https://docs.neptune.ai/usage/querying_metadata/#downloading-experiments-table-as-dataframe
     - https://docs.neptune.ai/tutorials/tracking_cross_validation_results/
     - https://docs.neptune.ai/tutorials/running_distributed_training/
     - https://docs.neptune.ai/tutorials/sequential_pipeline/
     - https://docs.neptune.ai/integrations/lightgbm/
     - https://docs.neptune.ai/tutorials/tracking_models_e2e/
   - Summarize how Neptune manages model metadata, which fields can be tracked (e.g., iterations, training loss, tuning parameters, feature importances, run parameters, hyperparameters, evaluation metrics), and the various integration points with tools like Optuna and XGBoost.
   - Detail any advice or best practices provided in these documents that might inform how we aggregate, filter, and visualize metadata.

2. **Extract and Compare Model Metadata for Runs sxm-449 and sxm-762:**
   
   - Focus on collecting all relevant metadata for two specific runs:
     - **sxm-449:** Note that it includes Optuna integration. Extract metadata fields that detail tuning iterations, hyperparameter selections, training results, and performance metrics.
     - **sxm-762:** Extract analogous metadata fields available for this run.
   - Use your internal documentation and code references only. Do not fabricate any data; if a metadata field is missing or its value is unclear, output “I don’t know” for that field.
   - Identify common metadata fields between these two runs, and note any differences (e.g., additional tuning fields in sxm-449 due to the Optuna integration).

3. **Determine Charting and Plotting Opportunities:**
   
   - Based on the metadata, identify which fields are available to be plotted. For example, chart iterations versus training loss, hyperparameter tuning progress, feature importance trends, and model evaluation metrics.
   - Describe the types of charts (e.g., line charts, scatter plots, bar charts) that would be most effective in comparing these runs.
   - Include examples of how the extracted metrics (iterations, training loss, tuning parameters, model features) could be visually represented. 
   - Reference guidelines from the Neptune documentation (provided URLs above) to support your suggestions for charting best practices.

4. **Sequential Analysis Using MCP and Memory Management:**
   
   - Leverage sequentialThinking mcp and mcp-memory server to break this analysis into clear, sequential subtasks:
     - (a) Analysis of Neptune documentation to identify chartable metadata fields.
     - (b) Extraction of metadata for sxm-449 and sxm-762.
     - (c) Comparison of metadata fields and determination of charting opportunities.
   - Preserve context between subtasks using your mcp-memory server, ensuring that all insights are integrated into the final output.

5. **Final Writeup and Integration into the Technical Walkthrough:**
   
   - Create a new Markdown section that details your findings from the analysis of documentation and metadata.
   - Incorporate detailed explanations, screenshots or image placeholders for relevant charts (if available internally), and code snippets that demonstrate how metadata is parsed and prepared for visualization.
   - Explain your choices for chart types and what insights traders can derive from them (e.g., how tuning improvements impact training performance).
   - Ensure the explanation references the internal source code or documentation lines whenever possible, and labels any uncertain or missing details with “I don’t know.”

6. **Output Requirements:**
   
   - The final output must be a single cohesive Markdown file that extends the existing parts.md. It should contain:
     - The metadata analysis writeup.
     - The comparative analysis for runs sxm-449 and sxm-762.
     - Detailed instructions for generating charts and plots.
     - A rationale for each visualization and integration, supported by internal code documentation.

Remember:

- Do not invent or infer any performance metrics or metadata values that are not directly evidenced in your internal documents.
- If any specific detail or field is ambiguous, explicitly indicate “I don’t know.”
- Use chain-of-thought reasoning and break your analysis into sequential subtasks for clarity and efficient token/memory management.
