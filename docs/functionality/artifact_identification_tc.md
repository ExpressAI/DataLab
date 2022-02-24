## 1. Observations and conclusions of artifact identification with PMI on the DBpedia2014

<table>
    <tr>
        <th width="60%">Observation</th>
        <th>Conclustion</th>
    </tr>
    <tr>
        <td>basicWord_{sent}&lt;0.444, PMI(label_{office holder}, basicWord_{sent})&gt;0.129;</td>
        <td rowspan="8">Sentences with more basic words (e.g. "is", "I") tend to be related to educational institution (the golden label is "educational institution".) and natural place, while sentences with fewer basic words tend to be related to "office holder", "artist", and "althlete".</td>
    </tr>
    <tr>
        <td>basicWord_{sent}&lt;0.37, PMI(label_{artist}, basicWord_{sent})&gt;0.138;</td>
        <td></td>
    </tr>
    <tr>
        <td>basicWord_{sent}&lt;0.37, PMI(label_{athlete}, basicWord_{sent})&gt;0.421;</td>
        <td></td>
    </tr>
    <tr>
        <td>basicWord_{sent}&gt;0.444, PMI(label_{office holder}, -basicWord_{sent})&gt;0.477;</td>
        <td></td>
    </tr>
    <tr>
        <td>basicWord_{sent}&gt;0.444, PMI(label_{company}, -basicWord_{sent})&gt;0.495;</td>
        <td></td>
    </tr>
    <tr>
        <td>basicWord_{sent}&gt;0.37, PMI(label_{village}, -basicWord_{sent})&gt;0.450;</td>
        <td></td>
    </tr>
    <tr>
        <td>basicWord_{sent}&gt;0.44, PMI(label_{artist}, -basicWord_{sent})&gt;0.402;</td>
        <td></td>
    </tr>
    <tr>
        <td>basicWord_{sent}&gt;0.44, PMI(label_{athlete}, -basicWord_{sent})&gt;0.952;</td>
        <td></td>
    </tr>
    <tr>
        <td>femaleWord_{sent}&gt;2.2, PMI(label_{artist}, femaleWord_{sent}) &gt;1.287</td>
        <td rowspan="2">Sentences with higher female bias are likely to be labeled "artist".</td>
    </tr>
    <tr>
        <td>femaleName_{sent}&gt;2.2, PMI(label_{artist}, femaleName_{sent}) &gt;0.326</td>
    </tr>
</table>

## 2. Observations and conclusions of artifact identification with PMI on the GLUE-qnli

<table>
    <tr>
        <th width="60%">Observation</td>
        <th>Conclustion</th>
    </tr>
    <tr>
        <td>len_{sent}&gt;35, PMI(label_{entailment}, len_{sent})&gt;0.023; </td>
        <td rowspan="2">Samples with long sentence are always labeled as "entailment", while samples with short sentences always belong to "not-entailment"  </td>
    </tr>
    <tr>
        <td>len_{sent}&gt;35, PMI(label_{not-entailment}, -len_{sent})&gt;0.034; </td>
        <td></td>
    </tr>
    <tr>
        <td>basicWord_{sent}&gt;0.5, PMI(label_{not-entailment}, basicWord_{sent})&gt;0.129; </td>
        <td rowspan="2">Sentences with fewer basic words are usually labeled as "entailment" labels, while samples containing more basic words are usually labeled as "not-entailment" labels.</td>
    </tr>
    <tr>
        <td>basicWord_{sent}&lt;0.5, PMI(label_{entailment}, basicWord_{sent})&gt;0.014; </td>
        <td></td>
    </tr>
    <tr>
        <td>readEase_{sent}&gt;37, PMI(label_{not-entailment}, readEase_{sent})&gt;0.021; </td>
        <td rowspan="2">Samples with easy-to-read sentences are usually ·entailment· relation, while samples with hard-to-read sentences are usually ·not-entailment·. </td>
    </tr>
    <tr>
        <td>readEase_{sent}&lt;37, PMI(label_{entailment}, readEase_{sent})&gt;0.045; </td>
        <td></td>
    </tr>
    <tr>
        <td>maleWord_{sent}&gt;6.3, PMI(label_{entailment}, maleWord_{sent}) &gt;0.518</td>
        <td rowspan="2">Sentences with higher male bias are likely to be labeled  ·entailment· .</td>
    </tr>
    <tr>
        <td>maleName_{sent}&gt;12.8, PMI(label_{entailment}, maleWord_{sent}) &gt;0.207</td>
        <td></td>
    </tr>
    <tr>
        <td>BLUE(que,sent)&gt;9.374, PMI(label_{entailment}, BLUE(que,sent)) &gt;0.5</td>
        <td>The higher the BLUE value of "question" and "sentence", samples are always labeled as ·entailment·.</td>
    </tr>
    <tr>
        <td>len_{que}/len_{sent}&gt;1.0, PMI(label_{entailment}, len_{que}/len_{sent}) &gt;0.046</td>
        <td>If the lengths of "question" is longer than "sentence", these samples are always labeled as ·entailment·.</td>
    </tr>
</table>