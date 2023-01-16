# IR-Project
<h2> Presenting </h2>

- &nbsp; Davit Shavit https://github.com/davis0011
- &nbsp; Noam Kent https://github.com/Kentno


<h3> Basic Preperation Overview </h3>

- &nbsp; Take Parquet files from bucket
- &nbsp; Use slightly modified version of assignment 3 gcp code to make indeciese
- &nbsp; Create a dict of metadata on each document, ex. full title with no tokenizing/stopword removal
- &nbsp; Transfer all data to an instance to run the retvival engine

<h3> Main Search Overview </h3>

All search functions other than "search" are according to the requirements.

- &nbsp; Search: implemented using a combination of BM25 on the body text, a count of the number of words in the title and a binary decision on anchor text.
- The scores are weighted 2-4-1, which was reached by testing severeal times on different parts of the training set.

- &nbsp; The main code block is the calculation of BM25 scores for each document in a numpy matrix, there is an explination of this in the report. See the graphic below:

