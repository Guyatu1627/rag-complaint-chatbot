Task 1:

Data Distribution and Volume Insights
Exploratory Data Analysis of the raw customer telemetry data revealed substantial imbalances across the customer feedback streams at CrediTrust Financial. Prior to pipeline execution, a significant portion of incoming complaints lacked detailed narrative context, serving only as categorical indicator flags. Among the four target service groups—Credit Cards, Personal Loans, Savings Accounts, and Money Transfers—complaints were heavily concentrated around Credit Card operations, reflecting higher daily processing velocities and payment exposure. Conversely, Savings Accounts exhibited the lowest comparative complaint frequency but carried higher baseline operational risks due to compliance and account-access disputes.

Narrative Structure and Volatility Profiling
Text distribution tracking metrics mapped across the Consumer complaint narrative fields revealed extreme volatility in length and depth. Narrative logs showed an expansive range, varying from single-sentence escalations (under 10 words) to highly detailed multi-page legal grievances exceeding 1,500 words. Short entries typically represented transactional errors with minimal structural context, whereas longer customer statements contained substantial noise, repetitive phrases, and nested structural timelines. This structural variance highlights the critical importance of implementing robust text-chunking strategies later in the RAG pipeline to prevent context truncation or vector space skewing.

Preprocessing and Text Normalization Strategy
To transform this noisy, unstructured customer feedback dataset into clean inputs optimized for vector search, we executed a rigorous data cleansing and normalization pipeline. Records outside the four primary financial products were filtered out, and entries without text narratives were dropped entirely. The remaining raw narratives were uniformly lowercased, and system placeholders or customer anonymization tokens (such as "XXXX") were systematically stripped out using regular expressions. Furthermore, standard operational boilerplate—including introductory expressions like "I am writing to file a complaint..."—was removed to ensure that our embedded vector database captures only dense, high-signal semantic details regarding actual financial anomalies.


### Task 2 Documentation: Sampling, Chunking, and Embedding Architecture

#### 1. Proportional Stratified Sampling Strategy
To maintain statistical parity with the complete consumer response stream, the raw database was downsampled to an optimal target array of 12,000 complaints using a proportional stratified sampling method. Grouping the datasets via the `Product` classification string allows the sampling grid to preserve identical channel density ratios (e.g., matching higher frequency volumes across Credit Cards against lower baseline volumes in Savings Accounts). This configuration ensures our RAG vector embeddings remain unskewed by majority class representation, providing Asha (Product Manager) with localized semantic responses matching true customer frequency vectors.

#### 2. Text Chunking Design Matrix Justification
Financial service narrative entries can be highly extensive and contain mixed problem categories within a single filing text block. To maximize embedding fidelity and prevent document-level semantic diluting, we implemented a `RecursiveCharacterTextSplitter` configured with a `chunk_size` of 500 characters and a `chunk_overlap` of 50 characters. 
* A size constraint of 500 characters maintains highly thematic text chunks, fitting within standard LLM semantic retrieval structures.
* The 50-character sliding overlap acts as a textual buffer, ensuring critical semantic topics or customer terms are not truncated across index partitions.

#### 3. Embedding Model Selection Framework
We integrated `sentence-transformers/all-MiniLM-L6-v2` as the core vector generation model. This specialized mini-language mapping transformer converts text string inputs into dense 384-dimensional spatial vectors. It is highly optimized for performance and production inference latencies, executing significantly faster than bulky alternative architectures while maintaining competitive accuracy. By enabling normalized embeddings (`normalize_embeddings=True`), lookup queries utilize stable dot-product calculations to output exact cosine similarity scores, matching user questions to relevant complaint fragments seamlessly.