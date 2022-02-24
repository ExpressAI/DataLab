# More Artifact Identification Examples in Popular Datasets



## 1. Observations and conclusions of artifact identification with PMI on the SNLI


<table>
    <tr>
        <th width="60%">Observation</th>
        <th width="40%">Conclustion</th>
    </tr>
    <tr>
        <td>len_{hp} &gt;8.4, PMI(label_{{neutral}}, len_{{hp}} )&gt;0.28;</td>
        <td>Long hypotheses tend to be neutral.</td>
    </tr>
    <tr>
        <td>len_{hp} \in [1,4.7], PMI(label_{{entailment}}, len_{{hp}} )=0.359;</td>
        <td>Short hypotheses tend to be entailment.</td>
    </tr>
    <tr>
        <td>flesch\_reading\_ease_{hp} \in [-50,1.352];</td>
        <td>When the hypothesis is difficult enough to read,</td>
    </tr>
    <tr>
        <td>PMI(label_{{entailment}}, {flesch\_reading\_ease}_{hp}) &gt;0.585;</td>
        <td>the sample tends to be labeled as entailment. </td>
    </tr>
    <tr>
        <td>male_{hp} &gt;2, PMI(label_{{neutral}}, male_{hp}) &gt;0.317;</td>
        <td rowspan="2">Hypotheses with gender bias words (male/female) tend to be neutral.</td>
    </tr>
    <tr>
        <td>female_{hp} &gt;2, PMI(label_{{neutral}}, male_{hp}) &gt;0.377;</td>
        <td></td>
    </tr>
    <tr>
        <td>X = len_{pm} - len_{hp}, if X \in [8,30];</td>
        <td rowspan="3">When the length difference of hypothesis and premise is small enough ([0,7]), the sample tends to be entailment, and when it is large enough ([8,30]) the sample tends to be entailment.</td>
    </tr>
    <tr>
        <td>PMI(label_{{entailment}}, len_{pm} - len_{hp}) &gt;0.084;</td>
        <td></td>
    </tr>
    <tr>
        <td>while X \in [0,7];  PMI(label_{{neutral}}, len_{pm} - len_{hp})=0.045;</td>
        <td></td>
    </tr>
    <tr>
        <td>X = len_{pm} + len_{hp}, if X  \in [4,13];</td>
        <td rowspan="3">When the sum of the lengths of hypothesis and premise is small enough,  the sample tends to be entailment, and when it is large enough it tends to be neutral.</td>
    </tr>
    <tr>
        <td>PMI(label_{{entailment}}, len_{pm} + len_{hp}) =0.259;</td>
        <td></td>
    </tr>
    <tr>
        <td>if X &gt;22, PMI(label_{{neutral}}, len_{pm} + len_{hp}) &gt;0.105; </td>
        <td></td>
    </tr>
    <tr>
        <td>X = len_{pm} / len_{hp}, if X &lt; 2,</td>
        <td rowspan="3">When the lengths of hypothesis and premise are close enough, the samples tend to be neutral, and when their lengths are sufficiently different, samples tend to be entailment.</td>
    </tr>
    <tr>
        <td>PMI(label_{{neutral}},len_{pm} / len_{hp}) &gt;0.094;</td>
        <td></td>
    </tr>
    <tr>
        <td>if X &gt; 2, PMI(label_{{entailment}},len_{pm} / len_{hp}) &gt;0.141;</td>
        <td></td>
    </tr>
    <tr>
        <td>PMI(label_{*},len_{pm}) \approx 0;</td>
        <td>The length and gender features of the premise are irrelevance  with the label.</td>
    </tr>
</table>

* `hp` and `pm` denote `hypothesis` and `premise`, respectively. `len` is a function that computes the length of a sentence. 


## 2. Observations and conclusions of artifact identification with PMI on the GLUE-SST2 
<table>
    <tr>
        <th width="60%">Observation</th>
        <th width="40%">Conclustion</th>
    </tr>
    <tr>
        <td>len_{sent} &lt;7, PMI(label_{positive}, len_{sent}) = 0.06</td>
        <td rowspan="2">Sentences that are long enough tend to be negative, while sentences that are short enough tend to be positive.</td>
    </tr>
    <tr>
        <td>len_{sent}  &gt;7,  PMI(label_{negative}, len_{sent}) &gt; 0</td>
        <td></td>
    </tr>
    <tr>
        <td>female_{sent}  \in [4.8,5.4], {PMI}({label}_{positive},{female}_{sent}) =0.58</td>
        <td rowspan="4">Sentences with low female bias tend to be negative, with high female bias tend to be positive; while sentences with high male bias tend to be negative.</td>
    </tr>
    <tr>
        <td>female_{sent}&lt;0.6, PMI(label_{negative},female_{sent}) =0.021</td>
        <td></td>
    </tr>
    <tr>
        <td>male_{sent}&lt;1.2, {PMI}({label}_{positive},{male}_{sent}) =0.018</td>
        <td></td>
    </tr>
    <tr>
        <td>male_{sent}&gt;1.2, PMI(label_{negative},male_{sent}) &gt;0.068</td>
    </tr>
</table>

* `len` is a function that computes the length of a sentence. `sent` denotes `sentence`.


## 3. Observations and conclusions of artifact identification with PMI on the DBpedia2014

<table>
    <tr>
        <th width="60%">Observation</th>
        <th width="40%">Conclustion</th>
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




## 4. Observations and conclusions of artifact identification with PMI on the GLUE-qnli

<table>
    <tr>
        <th width="60%">Observation</td>
        <th width="40%">Conclustion</th>
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

* `len` is a function that computes the length of a sentence. `sent` denotes `sentence`.

