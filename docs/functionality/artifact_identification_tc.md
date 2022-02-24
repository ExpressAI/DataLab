# Observations and conclusions of artifact identification with PMI on the DBpedia2014

| Observation                                                                  | Conclustion                                                          |
|------------------------------------------------------------------------------|----------------------------------------------------------------------|
| basicWord_{sent}<0.444, PMI(label_{office holder}, basicWord_{sent})>0.129;  | Sentences with more basic words (e.g.                                |
| basicWord_{sent}<0.37, PMI(label_{artist}, basicWord_{sent})>0.138;          | is, "I") tend to be related to educational                           |
| basicWord_{sent}<0.37, PMI(label_{athlete}, basicWord_{sent})>0.421;         | institution (the golden label is                                     |
| basicWord_{sent}>0.444, PMI(label_{office holder}, -basicWord_{sent})>0.477; | `educational institution`.) and natural                              |
| basicWord_{sent}>0.444, PMI(label_{company}, -basicWord_{sent})>0.495;       | place, while sentences with fewer basic                              |
| basicWord_{sent}>0.37, PMI(label_{village}, -basicWord_{sent})>0.450;        | words tend to be related to `office holder`,                         |
| basicWord_{sent}>0.44, PMI(label_{artist}, -basicWord_{sent})>0.402;         | `artist`, and `althlete`.                                            |
| basicWord_{sent}>0.44, PMI(label_{athlete}, -basicWord_{sent})>0.952;        |                                                                      |
| femaleWord_{sent}>2.2, PMI(label_{artist}, femaleWord_{sent}) >1.287         | Sentences with higher female bias are likely to be labeled `artist`. |
| femaleName_{sent}>2.2, PMI(label_{artist}, femaleName_{sent}) >0.326         |



# version 2

| Observation                                                                  | Conclustion                                                                                                                                                                                                                                                                |
|------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| basicWord_{sent}<0.444, PMI(label_{office holder}, basicWord_{sent})>0.129;  | Sentences with more basic words (e.g. "is", "I") tend to be related to educational institution (the golden label is `educational institution`.) and natural place, while sentences with fewer basic words tend to be related to `office holder`, `artist`, and `althlete`. |
| basicWord_{sent}<0.37, PMI(label_{artist}, basicWord_{sent})>0.138;          |                                                                                                                                                                                                                                                                            |
| basicWord_{sent}<0.37, PMI(label_{athlete}, basicWord_{sent})>0.421;         |                                                                                                                                                                                                                                                                            |
| basicWord_{sent}>0.444, PMI(label_{office holder}, -basicWord_{sent})>0.477; |                                                                                                                                                                                                                                                                            |
| basicWord_{sent}>0.444, PMI(label_{company}, -basicWord_{sent})>0.495;       |                                                                                                                                                                                                                                                                            |
| basicWord_{sent}>0.37, PMI(label_{village}, -basicWord_{sent})>0.450;        |                                                                                                                                                                                                                                                                            |
| basicWord_{sent}>0.44, PMI(label_{artist}, -basicWord_{sent})>0.402;         |                                                                                                                                                                                                                                                                            |
| basicWord_{sent}>0.44, PMI(label_{athlete}, -basicWord_{sent})>0.952;        |                                                                                                                                                                                                                                                                            |
| femaleWord_{sent}>2.2, PMI(label_{artist}, femaleWord_{sent}) >1.287         | Sentences with higher female bias are likely to be labeled `artist`.                                                                                                                                                                                                       |
| femaleName_{sent}>2.2, PMI(label_{artist}, femaleName_{sent}) >0.326         |
