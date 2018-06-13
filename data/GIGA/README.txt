####################
# GIGA Data Readme #
####################


### Files ###
- NYT_batches.txt: File segmenting up the filenames of the NYT portion of the Gigaword corpus into batches, zero-indexed. When the readme refers to batches, this file shows which NYT files those batches contain

- batch_0_ww_15000.csv.gz: Word-word co-occurrence matrix run on the zeroth batch of the NYT data, 15,000-word vocabulary

- batch_01_ww_15000_*.csv.gz: Word-word co-occurrence matrices from the first batch of the NYT data, 15,000-word vocabulary. This program was saving periodically but the code had a small bug so the files are named manually and it's unclear exactly which files were finished and saved. At most the first 4 files from the first batch were saved. I kept all files in case one or more of them caused problems. Any of these files can be combined with the batch_0 file because they will be disjoint, but these files should not be combined with each other because they do overlap.

- nyt5_15000.csv.gz: Run on the first 6 files of the NYT data, shouldn't combine with other files because it's not disjoint, but could use on its own for comparison


### Difficulties ###
- Had to work on AFS machines because of LDC data limitations
- Had trouble because of that because pipes kept breaking/connections kept timing out?
- Everything took a super long time to run - were going to try word-sentence co-occurrence matrices too, but it took even longer to run and the connection to AFS just kept timing out
- Tried submitting "jobs" on rice to run in the background but couldn't get it to work, potentially because of the specific packages/environment required, also data permissions? 
- It was just too late in the project to work out those difficulties, which was a shame for our project