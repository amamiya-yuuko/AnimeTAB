# AnimeTAB
A musicXML-based anime guitar tablature dataset

AnimeTAB is a musicXML-based guitar tablature dataset. All tracks of AnimeTAB are from anime songs and . Our dataset including three following parts:

## Full tracks
About 440 full tracks in musicXML format are in fulltracks folder, all tracks are derived from 'export to musicXML' function of Guitarpro7. 

## track clips
From 440 full tracks we choose 110 best quality tracks and cut them into 560 clips by different musical structures. The structure labels and begin/end bar numbers are labeled in file names. Four kinds of strucures are included: intro(I), verse(A), chorus(B) and bridge(C).

## encoded tracks
We encoded the pitches, finger positions and beats of every note clusters in measures into this format:
![Note encoding](https://github.com/amamiya-yuuko/AnimeTAB/blob/main/tokenized.png?raw=true)
by our TABgenerator pipeline. TABgenerator is an one-click data encoding tool especially for xml files exported by Guitarpro. We have also added somt other practical functions like 
key/chord detection, root/melody extraction, voices integration and so on. For more detial, please visit or paper on ISMIR2022:

# TABgenerator

To use TABgenerator, just:
```
import TABgenerator as TAB
pitchset, fingerset, timeset = TAB.readTAB(path)
```
it will encode all notes into

