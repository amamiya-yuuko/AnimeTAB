# AnimeTAB
AnimeTAB is a musicXML-based guitar tablature dataset. All tracks of AnimeTAB are arranged from anime and video game soundtracks. Our dataset including three following parts:

## Full tracks
About 400+ full tracks are in the fulltracks folder, all tracks are derived from 'export to musicXML' function of Guitarpro7. 

## track clips
From 440 full tracks we choose 200+ best quality tracks and cut them into 560 clips by different musical structures. The structure labels and begin/end bar numbers are labeled in file names. Four kinds of strucures are included: intro(I), verse(A), chorus(B) and bridge(C).

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

This function recognize a chord's type by the in-chord intervals. For now it supports:

```
    '''
    maj: interval loops as 435 435 435 or 75 or 48 or 39 semitones
    min: 345 345 345
    maj7: ， 4341 4341 4341 or 471
    m7: 3432 3432 3432 or 732
    7:  4332 4332 4332
    aug: 444 444 444
    dim: 326 326 326
    '''
```

## Convert between MIDI value/encoded pitch/finger positions
```
str2midi(str1)
midi2str(midivalue)
finger2midi(finger)
midi2finger(midivalue)
```
Special tunings also supported

## Root and melody extraction
```
root_detect(measure)
```
## xml files output(unstable, editing...)
```
generateTAB(pitch, finger, time)
```

################

Visit our paper on ISMIR2022 for more detials:

Contact us:Yuecheng_Zhou@cuc.edu.cn
