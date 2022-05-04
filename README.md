# AnimeTAB
AnimeTAB is a musicXML-based guitar tablature dataset. All tracks of AnimeTAB are from anime songs and . Our dataset including three following parts:

## Full tracks
About 440 full tracks in musicXML format are in fulltracks folder, all tracks are derived from 'export to musicXML' function of Guitarpro7. 

## track clips
From 440 full tracks we choose 110 best quality tracks and cut them into 560 clips by different musical structures. The structure labels and begin/end bar numbers are labeled in file names. Four kinds of strucures are included: intro(I), verse(A), chorus(B) and bridge(C).

## encoded tracks
We encoded the pitches, finger positions and beats of every note clusters in measures into this format:
![Note encoding](https://github.com/amamiya-yuuko/AnimeTAB/blob/main/tokenized.png?raw=true)
by our TABgenerator pipeline. TABgenerator is an one-click data encoding tool especially for xml files exported by Guitarpro. We have also added somt other practical functions like 
key/chord detection, root/melody extraction, voices integration and so on. 

# TABgenerator

To use TABgenerator, just:
```
import TABgenerator as TAB
pitchset, fingerset, timeset = TAB.readTAB(path)
```
Other functions including:

## Special events detect
```
skills_detect(noteNode)
```
↑This function return a dict of whether there is a possible special event in this node. For now it supports:
```
special_skills = {        
        'pitch':False, #if it has a pitch
        'duration':False, #if it has a duration
        'rest':False, #if it is a rest
        'dot':False, #if it has a dot
        'tie':True, 
        'chord':False, #if this note is sound at the same time as former note
        'tuplet':False,  # -3-
        'grace':False,  
        'artificial':False,   #artificial harmonic
        'natural':False, #natural harmonic
        'mute':False  #if it's a dead note
    }
```
and users can add the dict for more special events

## Key detection and shift

```
key_detect(pitchlist)
key_shift(origin_key, new_key)
```

key_detect function suspect the key by the distribution of all pitches. 

## Chord recognition
```
chord_recognize(pitch_cluster)
```

This function racognize a chord's type by the in-chord intervals. For now it supports:

```
    '''
    maj：有且仅有大三度和五度， 435 435 435 或 75 或 48 39
    min：有且仅有小三度和五度， 345 345 345
    maj7：存在大七度和大三度， 4341 4341 4341或471
    m7：存在小七度和小三度， 3432 3432 3432或732
    7：存在小七度和大三度， 4332 4332 4332或732
    aug：存在增五度（小六度）， 444 444 444
    dim：存在增四度， 326 326 326或
    '''
```

## 
For more detial, please visit or paper on ISMIR2022:

Contact us:Yuecheng_Zhou@cuc.edu.cn