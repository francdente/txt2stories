Inter-annotator agreement data referenced in Section V of the paper.

- `Inter_annotation_agreement.csv`: 50 chunk-story pairs annotated by three annotators, used to compute Fleiss' kappa.
- `g01_user_stories.csv`, `g04_user_stories.csv`: User stories from Student Projects 1 and 2.
- `random_user_story_draw.py`: Script used to randomly sample user stories for the inter-annotation experiment.

**Inter-annotator agreement process**

To evaluate the consistency of the annotations, we measured the inter-annotator agreement on a subset of the annotated datasets.  
The first author initially annotated all chunk–story pairs following the protocol described in Section V.  
To evaluate reliability, we randomly selected five user stories from the two student projects used for manual annotation (Student 1 and Student 2).  
Random sampling was performed using a fixed seed, and the exact selection script is included.

For each of the ten selected user stories (five per project), we extracted the corresponding chunks previously annotated by the first author and submitted them for re-annotation to the two co-authors, who had not participated in the initial dataset annotation.  
Each annotator independently judged whether each chunk supported the story (\texttt{1}) or not (\texttt{0}).

We then computed Fleiss’~$\kappa$ across the three annotators.  
The overall agreement is $\kappa{=}0.470$, which corresponds to a \emph{moderate} level of agreement according to standard interpretation scales.  

The CSV files containing the full set of survey responses (10 user stories, 5 per project) are available in this folder (See top of this file).
