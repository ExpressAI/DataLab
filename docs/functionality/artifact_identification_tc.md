## Observations and conclusions of artifact identification with PMI on the DBpedia2014


| Observation                                                                  | Conclustion                                   |
|------------------------------------------------------------------------------|-----------------------------------------------|
| basicWord_{sent}<0.444, PMI(label_{office holder}, basicWord_{sent})>0.129;  | Sentences with more basic words (e.g.         |
| basicWord_{sent}<0.37, PMI(label_{artist}, basicWord_{sent})>0.138;          | is, "I") tend to be related to educational    |
| basicWord_{sent}<0.37, PMI(label_{athlete}, basicWord_{sent})>0.421;         | institution (the golden label is              |
| basicWord_{sent}>0.444, PMI(label_{office holder}, -basicWord_{sent})>0.477; | `educational institution`.) and natural       |
| basicWord_{sent}>0.444, PMI(label_{company}, -basicWord_{sent})>0.495;       | place, while sentences with fewer basic       |
| basicWord_{sent}>0.37, PMI(label_{village}, -basicWord_{sent})>0.450;        | words tend to be related to `office holder`,  |
| basicWord_{sent}>0.44, PMI(label_{artist}, -basicWord_{sent})>0.402;         | `artist`, and `althlete`.                     |
| basicWord_{sent}>0.44, PMI(label_{athlete}, -basicWord_{sent})>0.952;        |                                               |
| femaleWord_{sent}>2.2, PMI(label_{artist}, femaleWord_{sent}) >1.287         | Sentences with higher female bias are likely  |
| femaleName_{sent}>2.2, PMI(label_{artist}, femaleName_{sent}) >0.326         | to be labeled `artist`.                       |


## Observations and conclusions of artifact identification with PMI on the GLUE-qnli

| Observation                                                                    | Conclustion                                                                         |
|--------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| len_{sent}>35, PMI(label_{entailment}, len_{sent})>0.023;                      | Samples with long sentence are always labeled as `entailment`,                      |
| len_{sent}>35, PMI(label_{not-entailment}, -len_{sent})>0.034;                 | while samples with short sentences always belong to `not-entailment`.               |
| basicWord_{sent}>0.5, PMI(label_{not-entailment}, basicWord_{sent})>0.129;     | Sentences with fewer basic words are usually labeled as `entailment` labels, while  |
| basicWord_{sent}<0.5, PMI(label_{entailment}, basicWord_{sent})>0.014;         | samples containing more basic words are usually labeled as `not-entailment` labels. |
| readEase_{sent}>37, PMI(label_{not-entailment}, readEase_{sent})>0.021;        | Samples with easy-to-read sentences are usually ·entailment· relation, while        |
| readEase_{sent}<37, PMI(label_{entailment}, readEase_{sent})>0.045;            | samples with hard-to-read sentences are usually ·not-entailment·.                   |
| maleWord_{sent}>6.3, PMI(label_{entailment}, maleWord_{sent}) >0.518           | Sentences with higher male bias are likely to be labeled  ·entailment·.             |
| maleName_{sent}>12.8, PMI(label_{entailment}, maleWord_{sent}) >0.207          |                                                                                     |
| BLUE(que,sent)>9.374, PMI(label_{entailment}, BLUE(que,sent)) >0.5             | The higher the BLUE value of `question` and `sentence`,                             |
|                                                                                | samples are always labeled as ·entailment·.                                         |
| len_{que}/len_{sent}>1.0, PMI(label_{entailment}, len_{que}/len_{sent}) >0.046 | If the lengths of `question` is longer than                                         |
|                                                                                | `sentence`, these samples are always labeled as ·entailment·.                       |


<table>
    <tr>
        <td>Observation</td>
        <td>Conclustion</td>
    </tr>
    <tr>
        <td>len_{sent}&gt;35, PMI(label_{entailment}, len_{sent})&gt;0.023; </td>
        <td>Samples with long sentence are always labeled as `entailment`, while samples with short sentences always belong to `not-entailment`  </td>
    </tr>
    <tr>
        <td>len_{sent}&gt;35, PMI(label_{not-entailment}, -len_{sent})&gt;0.034; </td>
        <td></td>
    </tr>
    <tr>
        <td>basicWord_{sent}&gt;0.5, PMI(label_{not-entailment}, basicWord_{sent})&gt;0.129; </td>
        <td>Sentences with fewer basic words are usually labeled as `entailment` labels, while samples containing more basic words are usually labeled as `not-entailment` labels.</td>
    </tr>
    <tr>
        <td>basicWord_{sent}&lt;0.5, PMI(label_{entailment}, basicWord_{sent})&gt;0.014; </td>
        <td></td>
    </tr>
    <tr>
        <td>readEase_{sent}&gt;37, PMI(label_{not-entailment}, readEase_{sent})&gt;0.021; </td>
        <td>Samples with easy-to-read sentences are usually ·entailment· relation, while samples with hard-to-read sentences are usually ·not-entailment·. </td>
    </tr>
    <tr>
        <td>readEase_{sent}&lt;37, PMI(label_{entailment}, readEase_{sent})&gt;0.045; </td>
        <td></td>
    </tr>
    <tr>
        <td>maleWord_{sent}&gt;6.3, PMI(label_{entailment}, maleWord_{sent}) &gt;0.518</td>
        <td>Sentences with higher male bias are likely to be labeled  ·entailment· .</td>
    </tr>
    <tr>
        <td>maleName_{sent}&gt;12.8, PMI(label_{entailment}, maleWord_{sent}) &gt;0.207</td>
        <td></td>
    </tr>
    <tr>
        <td>BLUE(que,sent)&gt;9.374, PMI(label_{entailment}, BLUE(que,sent)) &gt;0.5</td>
        <td>The higher the BLUE value of `question` and `sentence`, samples are always labeled as ·entailment·.</td>
    </tr>
    <tr>
        <td>len_{que}/len_{sent}&gt;1.0, PMI(label_{entailment}, len_{que}/len_{sent}) &gt;0.046</td>
        <td>If the lengths of `question` is longer than `sentence`, these samples are always labeled as ·entailment·.</td>
    </tr>
</table>