#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from xml.dom import minidom
from xml.dom.minidom import parse
import os

global gl_tune
global gl_capo
gl_tune = [0, 0, 0, 0, 0, 0]
gl_capo = 0

# In[17]:
def getAttr(rootNode, childNode):
    return rootNode.getElementsByTagName(childNode)[0].firstChild.data

def haveNodes(rootNode, childNode):
    childNodelist = rootNode.getElementsByTagName(childNode)
    len1 = len(childNodelist)
    if len1 != 0:
        #if chilNodelist.getAttribute()
        return True
    elif len1 == 0:
        return False
    
def skills_detect(noteNode):   
    #******detect special skills in XML******
    special_skills = {
        'pitch':False,
        'duration':False,
        'rest':False,
        'dot':False,
        'tie':False,
        'chord':False,
        'tuplet':False,  #三联音
        'grace':False,  #装饰音
        'artificial':False,   #人工泛音
        'natural':False, #自然泛音
        'mute':False  #闷音 就是那个x
    }
    for skill in special_skills.keys():
        special_skills[skill] = haveNodes(noteNode, skill)
    return special_skills
    
#修改后的xml写入函数
def fixed_writexml(self, writer, indent="", addindent="", newl=""):  
    # indent = current indentation  
    # addindent = indentation to add to higher levels  
    # newl = newline string  
    writer.write(indent+"<" + self.tagName)  

    attrs = self._get_attributes()  
    a_names = attrs.keys()  
    sorted(a_names)  

    for a_name in a_names:  
        writer.write(" %s=\"" % a_name)  
        minidom._write_data(writer, attrs[a_name].value)  
        writer.write("\"")  
    if self.childNodes:  
        if len(self.childNodes) == 1 and self.childNodes[0].nodeType == minidom.Node.TEXT_NODE:  
            writer.write(">")  
            self.childNodes[0].writexml(writer, "", "", "")  
            writer.write("</%s>%s" % (self.tagName, newl))  
            return  
        writer.write(">%s"%(newl))  
        for node in self.childNodes:  
            if node.nodeType is not minidom.Node.TEXT_NODE:  
                node.writexml(writer,indent+addindent,addindent,newl)  
        writer.write("%s</%s>%s" % (indent,self.tagName,newl))  
    else:  
        writer.write("/>%s"%(newl))
minidom.Element.writexml = fixed_writexml #换掉包里的原函数


# In[18]:


#根据原xml文件取文件头的 大部分特殊调弦信息、时间信息等都在文件头中，所以直接把原文件头拿来用
def make_xmlhead(file):
    domtree = parse(file)
    
    measureNode = domtree.getElementsByTagName('measure')
    measurenum = len(measureNode)
    for i in range(measurenum - 1):
        measureNode = domtree.getElementsByTagName('measure')[-1]
        measureNode.parentNode.removeChild(measureNode)
    noteNodes = domtree.getElementsByTagName('note')
    for Node in noteNodes:
        Node.parentNode.removeChild(Node)
    backupNodes = domtree.getElementsByTagName('backup')
    for Node in backupNodes:
        Node.parentNode.removeChild(Node)
    
    name = file.split('\\')[-1]
    newfilepath = file.split('\\')
    newfilepath = '\\'.join(newfilepath[:-1]) + '\\head_{}.xml'.format(name)
    with open(newfilepath, 'w', encoding='utf-8') as f:
        domtree.writexml(f, addindent=' ', newl='\n')
    return newfilepath

def choose_measure(songs, song_num, measure_nums, mode='notes'):
    song_index = songs[song_num]
    #song_index的结构太难背了 总记不住写个选小节的函数吧
    choosen = []
    if mode == 'notes':
        for num in measure_nums:
            choosen.append(song_index[0][num])
        
    if mode == 'fingers':
        for num in measure_nums:
            choosen.append(song_index[1][num])  

    if mode == 'times':
        for num in measure_nums:
            choosen.append(song_index[2][num])
    
    return choosen


def get_parts(measure):
    measurechilds = [i for i in measure.childNodes if (hasattr(i, 'tagName') and (i.tagName == 'note' or i.tagName == 'backup'))]
    total_childs = len(measurechilds)
    backupindex = len(measure.getElementsByTagName('backup')) //2
    if len(measure.getElementsByTagName('backup')) == 0:
        return [measurechilds]
    else:
        all_backup_pos = [i for i, x in enumerate(measurechilds) if x.tagName == 'backup']

        real_backupindex = all_backup_pos[backupindex]

        former_backup = all_backup_pos[all_backup_pos.index(real_backupindex):]
        former_backup.append(total_childs)
        parts = []
        for i in range(len(former_backup)):
            if i == len(former_backup)-1:
                break
            part = [i for i in measurechilds[former_backup[i]:former_backup[i+1]+1] if i.tagName != 'backup']
            parts.append(part)
        return parts
  
def part_integrate(parts):
    '''
    要求：parts为[dict part1, dict part2, dict part3]
    每个dict part拥有dictpitch, dictfinger, dicttime三个key
    '''
    total_cluster = []
    total_times = []
    for partdict in parts:
        cumtime0 = np.cumsum(partdict['dicttime']) - partdict['dicttime']
        total_times.append(list(cumtime0))
        cluster0 = list(zip(partdict['dictpitch'], partdict['dictfinger'], cumtime0))
        total_cluster.extend(cluster0)
        
    total_times = sorted(set(split(total_times)))
    pitch_cus, finger_cus = [], []
    for q in total_times:
        _ = [i for i in total_cluster if i[2]==q]
        pitch_list, finger_list = [i[0] for i in _], [i[1] for i in _]
        pitch_cu, finger_cu = '', ''
        for _pitch, _finger in zip(pitch_list, finger_list):
            pitch_cu += (_pitch + ' ')
            finger_cu += (_finger + ' ')
        pitch_cus.append(pitch_cu[:-1])
        finger_cus.append(finger_cu[:-1])
    return [pitch_cus, finger_cus, total_times]
  
# In[20]:
divisions = 0
def timesig_detect(measure, divisions, beats=4):
    timesig = []
    #******小节每拍长度******
    #******小节总时长******
    noteNodes = measure
    #print('note num in timesig_detect:{}'.format(len(noteNodes)))
    # if haveNodes(measure, 'backup'):
    #     noteNodes = noteNodes[:len(noteNodes)//2]  #获取所有note节点 只取前一半 因为有个backup会把所有节点重复一次 所以只去一半就可
    Nodesnum = len(noteNodes)
    total_duration = 0
    for i, Node in enumerate(noteNodes):
        if haveNodes(Node, 'chord') == False:
            if haveNodes(Node, 'duration') == True:
                total_duration += int(getAttr(Node, 'duration'))


    
    total_duration = divisions * int(beats)
    #print('小节总duration:{}'.format(total_duration))
    #measure_num = measure.getAttribute('number')

    notes = measure
    # if haveNodes(measure, 'backup'):
    #     notes = notes[:len(notes)//2]
    notenum = len(notes)
    for i, note in enumerate(notes):
        #print('notenum={}, note No.{}processing...'.format(Nodesnum, i+1))
        #******detect special skills in XML******
        special_skills = skills_detect(note)
        #******detect time******
        if special_skills['grace'] == True:
            timesig.append(0)
        if special_skills['duration'] == True:
            if special_skills['chord'] == False:
                duration = getAttr(note, 'duration')
                notetype = getAttr(note, 'type')
                timesus = int(duration) / divisions
                timesig.append(timesus)
    return timesig

def pitch_detect(measure):
    notes = measure
    #print('note num in timesig_detect:{}'.format(len(notes)))
    notenum = len(notes)
    #******侦测音高******
    notestep = []
    for i, note in enumerate(notes):
        skills = skills_detect(note)
        if ((skills['pitch'] == True or skills['chord'] == True) and skills['artificial'] == False) or (skills['artificial'] == True and skills['chord']== False):
            pitchs = note.getElementsByTagName('pitch')
            steps = pitchs[0].getElementsByTagName('step')
            alters = pitchs[0].getElementsByTagName('alter')
            octaves = pitchs[0].getElementsByTagName('octave')
            step0 = steps[0].firstChild.data
            alt = ''
            if len(alters) > 0:
                altchange = alters[0].firstChild.data
                if altchange == '1':
                    alt = '#'
                elif altchange == '-1':
                    alt = '-'
                  
                
            step0 = str(octaves[0].firstChild.data) + steps[0].firstChild.data + alt
            #print('step0:{}'.format(step0))
            chordtree = []
            if skills['pitch'] and skills['chord'] == False:
                notestep.append(step0)
            elif skills['chord']:
                notestep[-1] = notestep[-1] + ' ' + step0
        elif note.childNodes[1].nodeName == 'rest':
            rests = note.getElementsByTagName('rest')
            notestep.append('R')  
    return notestep

def finger_detect(measure, capo=0):
    fingers = []
    notes = measure
    #print('note num in timesig_detect:{}'.format(len(notes)))
    notenum = len(notes)
    for i, note in enumerate(notes):
        #print('note {}'.format(i))
        skills = skills_detect(note)
        if skills['rest'] == False and skills['artificial'] == False:
            strings = note.getElementsByTagName('string')
            frets = note.getElementsByTagName('fret')
            string = strings[0].firstChild.data
            fret = frets[0].firstChild.data
            finger = '({},{})'.format(string, fret)
            if skills['pitch'] and skills['chord'] == False:
                fingers.append(finger)
            elif skills['chord']:
                fingers[-1] = fingers[-1] + ' ' + finger       
        elif skills['rest'] == True:
            fingers.append('(R,R)')
        elif skills['artificial'] == True and skills['chord'] == False:
            pitch = getAttr(note, 'octave') + getAttr(note, 'step')
            if haveNodes(note, 'alter'):
                altchange = note.getElementsByTagName('alter')[0].firstChild.data
                if altchange == '1':
                    pitch += '#'
                elif altchange == '-1':
                    pitch += '-'
            #print('note {}, {}'.format(i, pitch))
            fingers.append(midi2finger(str2midi(pitch), tune=gl_tune, capo=capo)[-1])
    return fingers

def tie_clean(measure, divisions=divisions, beats=4, capo=gl_capo):
    #******获得所有noteNode节点******
    #print('measure {} processing...'.format(measure_num))
    #noteNodes = measure.getElementsByTagName('note')
    noteNodes = measure
    # if haveNodes(measure, 'backup'):
    #     noteNodes = noteNodes[:len(noteNodes)//2]
    Nodesnum = len(noteNodes)
    #print('{} nodes in measure'.format(Nodesnum))
    #******获得所有noteNode的tie情况，存放于tietypes中******
    tietypes = []
    for noteNode in noteNodes:
        tietype = []
        if haveNodes(noteNode, 'tie'):          
            for tieNode in noteNode.getElementsByTagName('tie'):
                tietype.append(tieNode.getAttribute('type'))

        tietypes.append(tietype)
    #******获得timesig pitchs fingers******
    timesig = timesig_detect(measure, divisions=divisions, beats=beats)
    pitch_cus = pitch_detect(measure)
    finger_cus = finger_detect(measure, capo=capo)
    # print('before del fingers:{}'.format(finger_cus))
    # print('timesig:{}'.format(timesig))
    # print('pitch_cus:{}'.format(pitch_cus))
    pitchs = []
    for pitch in pitch_cus: 
        pitchs.append(pitch.split())
    #******timesig中每个音簇只选取一个 其余的删掉******
    new_timesig = timesig
    total_pitch_count = -1
    for i in range(len(pitch_cus)):
        if i < len(pitch_cus) - 1:
            #print('第{}循环取对'.format(i))
            before_cu = pitch_cus[-i-2].split()
            after_cu = pitch_cus[-i-1].split()
            before_fingercu = finger_cus[-i-2].split()
            after_fingercu = finger_cus[-i-1].split()
            # print('before_cu:{}'.format(before_cu))
            # print('after_cu:{}'.format(after_cu))
            ifnoties = 1
            before_cu_start = total_pitch_count - len(before_cu) - len(after_cu) + 1
            before_cu_end = total_pitch_count - len(after_cu)

            after_cu_start = before_cu_end + 1
            after_cu_end = total_pitch_count
            total_pitch_count -= len(after_cu) 
            #print('将遍历从{}到{}'.format(before_cu_start, before_cu_end))
            for q in np.arange(abs(before_cu_end), abs(before_cu_start)+1):
                if 'start' in tietypes[-q]:
                    ifnoties = 0

            if ifnoties == 1:
                continue

            ifcover = 1  #先假设可以cover 即后面簇的每个音都是tiestop
            for s in np.arange(abs(after_cu_end), abs(after_cu_start)+1):
                if 'stop' not in tietypes[-s]:
                    ifcover = 0

            if ifcover == 0: #当后面簇并不是每个音都是tiestop时：直接把tiestop的音删除
                after_cu_tie = []
                for q2 in np.arange(abs(after_cu_end), abs(after_cu_start)+1):
                    if 'stop' in tietypes[-q2]:
                        after_cu_tie.append(True)
                    else:
                        after_cu_tie.append(False)
                after_cu_tie = list(reversed(after_cu_tie))
                # print('after_cu_tie:{}'.format(after_cu_tie))


                for _ in range(len(after_cu_tie)):
                    if after_cu_tie[_]:
                        after_cu[_], after_fingercu[_] = '', ''
                        
                after_cu_new, after_fingercu_new = '', ''
                for _2 in zip(after_cu, after_fingercu):
                    after_cu_new += (_2[0] + ' ')
                    after_fingercu_new += (_2[1] + ' ')
                pitch_cus[-i-1] =  after_cu_new.strip()  #strip()用于删去前后多余空格
                finger_cus[-i-1] =  after_fingercu_new.strip()


            if ifcover == 1: #当后面簇每个音都是tiestop时 删除后面所有音并把时值加到前面去
                pitch_cus[-i-1] = ''
                finger_cus[-i-1] = ''
                new_timesig[-i-2] += new_timesig[-i-1]
                new_timesig[-i-1] = 0
    #print(pitch_cus)    
    while '' in pitch_cus:
        pitch_cus.remove('')
    while '' in finger_cus:
        finger_cus.remove('')
    while 0 in new_timesig:
        new_timesig.remove(0)
    # print('###########')
    # print(new_timesig)
    # print(pitch_cus)
    # print('###########')
    # print('measure {} done'.format(measure_num), end='\n\n')
    return [new_timesig, pitch_cus, finger_cus]

def str_postprocess(pitchstr, mode='pitch', pitchtype='default', capo=0):
    if mode == 'pitch':
        new_cu = ''
        pitchs = [str2midi(i) for i in pitchstr.split()]
        alter = '#'
        if '-' in pitchstr:
            alter = '-'
        newlist = [midi2str(j, alter=alter) for j in sorted(list(filter(None, pitchs)), reverse=True)]
        for pitch in newlist:
            if pitchtype == 'music21':
                pitch = pitch[1:] + pitch[0]
            new_cu += (pitch + ' ')
        if new_cu == '':
            return 'R'
        else:
            return new_cu[:-1]
    
    if mode == 'finger':
        new_cu = ''
        pitchs = [[i, finger2midi(i, tune=gl_tune, capo=capo)] for i in pitchstr.split() if i != '(R,R)']
        newlist = sorted(pitchs, key = lambda x: x[1])
        newlist = reversed(newlist)
        for pitch in newlist:
            new_cu += (pitch[0] + ' ')
        if new_cu == '':
            return '(R,R)'
        else:
            return new_cu[:-1]


class TABinfo:
    def __init__(self, beat=4, beat_type=4, capo=0, tuning=[0,0,0,0,0,0], original=None, name=None):
        self.beat = beat
        self.beat_type = beat_type
        self.capo = capo
        self.tuning = tuning
        self.original = original
        self.name = name
    def showinfo(self):
        infodict = {
            'beat': int(self.beat),
            'beat_type': int(self.beat_type),
            'capo': self.capo,
            'tuning': self.tuning,
            'original': self.original,
            'name': self.name
        }
        print(infodict)
class Tablature:
    def __init__(self, pitch, finger, time, info=None):
        self.pitch = pitch #该歌曲的音符，每个小节为一个list，所有小节组合成一个大list
        self.finger = finger #指法
        self.time = time #时值
        self.info = info
    def choose_measure(self, measurenum):
        '''
        check the given measure number()
        =============================
        measurenum: int or list
        the measure number you want to check
        '''
        if isinstance(measurenum, int):
            measure = Tablature(pitch=self.pitch[measurenum],
                                finger=self.finger[measurenum],
                                time=self.time[measurenum],
                                info=self.info
                               )
            return measure
        elif isinstance(measurenum, (list, tuple)):
            measure = Tablature(pitch=self.pitch[measurenum[0]:measurenum[1]],
                                finger=self.finger[measurenum[0]:measurenum[1]],
                                time=self.time[measurenum[0]:measurenum[1]],
                                info = self.info)
            return measure
        else:
            raise Exception('Input Error: input should be int or list, but {}'.format(measurenum, type(measurenum)))
    def vectorization(self, division=16):
        measurenum = len(self.pitch)
        measurelist = []
        for i in range(measurenum):
            measurelist.append(self.choose_measure(i))
        
        pitchvec, fingervec = [], []
        for measure in measurelist:
            info = measure.info #
            beat = info.beat
            pitchseries, fingerseries= [], []
            for i, index in enumerate(zip(measure.pitch, measure.finger, measure.time)):
                note = []
                finger = []
                for i in range(int(index[2] * (division/4))):
                    if i == 0:
                        note.append(index[0])
                        finger.append(index[1])
                    else:
                        note.append('SUS')
                        finger.append('SUS')

                for notezip in zip(note, finger):
                    pitchseries.append(notezip[0])
                    fingerseries.append(notezip[1])
            pitchvec.append(pitchseries)
            fingervec.append(fingerseries)
        vecsong = Tablature(pitch=pitchvec, finger=fingervec, time=self.time, info=self.info)
        return vecsong
    
    def key(self, return_vector=False):
        '''
        pitchlist: a list of measures, every measure is a list of clusters 
        '''
        pitchlist = self.pitch
        all_pitchs = split(pitchlist)
        key_vector = np.zeros(12)#调性向量，物理意义是不同音出现的次数

        #pitch_dict:12个音
        pitch_dict = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        minors_dict = ['a', 'a#', 'b', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#']
        double_dict = pitch_dict.copy()
        double_dict.extend(double_dict)
        intervals = [2, 2, 1, 2, 2, 2] #全全半全全全半
        intervals = np.array(intervals).cumsum()


        for pitch in all_pitchs:
            add_vector = np.zeros(12)
            singles = pitch.split()
            for single in [q for q in singles if q != 'R']:
                #single = single[1:]
                midivalues_index = str2midi(single) % 12
                add_vector[midivalues_index] += 1
                # if single in pitch_dict:
                #     add_vector[pitch_dict.index(single)] += 1   
            key_vector += add_vector
        all_keys = []
        for start in pitch_dict:
            add_num = pitch_dict.index(start)
            Ckey = [0, 2, 4, 5, 7, 9, 11]

            new = [i+add_num for i in Ckey]
            for i in range(len(new)):
                if new[i] >= 12:
                    new[i] -= 12
            all_keys.append(new)

        key_sums = []
        for key in all_keys:
            key_sums.append(sum([key_vector[position] for position in key]))

        maj_key = pitch_dict[np.argmax(key_sums)]
        min_key = minors_dict[np.argmax(key_sums)]

        majsum, minsum = 0, 0
        for measure in pitchlist:
            firstcluster = []
            for cu in measure:
                if ' ' in cu:
                    firstcluster.append(cu)
                    break

            for cluster in firstcluster:            
                inpitch = [i[1:] for i in cluster.split()]
                #print("**{}: {}**".format(cluster, cutype))
                if double_dict[pitch_dict.index(maj_key) + 7] in inpitch:
                    majsum += 1
                if min_key.upper() in inpitch:
                    minsum += 1

        if majsum >= minsum:
            real_key = maj_key
        else:
            real_key = min_key

        print('maj {}, min {}'.format(majsum, minsum))
        if return_vector:
            return real_key, key_vector
        else:
            return real_key
          
    def key_shift(self, new_key):
        '''
        给小节移调的
        '''
        origin_key = self.key()
        pitch_dict = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        minors_dict = ['a', 'a#', 'b', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#']
        if origin_key in pitch_dict:
            ori = pitch_dict.index(origin_key)
        elif origin_key in minors_dict:
            ori = minors_dict.index(origin_key)

        if new_key in pitch_dict:
            neww = pitch_dict.index(new_key)
        elif new_key in pitch_dict:
            neww = minors_dict.index(new_key)

        all_pitchs = []    
        for pitchlist in self.pitch:
            delta_midi = neww - ori
            new_measure = []
            for pitch_cu in pitchlist:
                pitchs = pitch_cu.split()
                new_pitchs = []
                for pitch in pitchs:
                    if pitch != 'R':
                        new_pitchs.append(midi2str(str2midi(pitch) + delta_midi))
                    else:
                        new_pitchs.append('R')

                new_cu = ''
                for pitch in new_pitchs:
                    new_cu += pitch
                    new_cu += ' '
                new_measure.append(new_cu)
            all_pitchs.append(new_measure)

        return all_pitchs
    def root_and_melody(self, clear_zero=True):
        '''
        根音和旋律音检测
        measure:一个三个元素的list，分别是小节内的所有音、指法、时间
        '''
        
        measurenum = len(self.pitch)
        measurelist = []
        for i in range(measurenum):
            measurelist.append(self.choose_measure(i))
        
        readymeasures = []
        for measure in measurelist:
            
            pitchs, fingers, times = measure.pitch, measure.finger, measure.time #分别代表音高、指法、时间
            tune, capo = measure.info.tuning, measure.info.capo
            pitchnum = len(pitchs)
            indexlist = list(np.zeros(pitchnum))
            #print('待检测的音为{}'.format(pitchs))
            roots = [] 
            melodys = []
            root_thr = 5

            #设置根音门槛
            allfingers = []
            for finger in fingers:
                allfingers.extend(finger.split())
            rootcount = 0
            for finger in allfingers:
                if ('R' not in finger) and int(finger[1]) >= 5:
                    rootcount += 1
            if rootcount <= 1:
                root_thr = 4


            for i in range(pitchnum):  #该小节内的第i个音

                multi = 0
                pitch_cu = pitchs[i].split()
                finger_cu = fingers[i].split()
                if len(pitch_cu) >= 2:
                    multi = 1
                melody, root = 0, 0
                if 'R' not in finger_cu[0]:
                    for k in range(len(finger_cu)):
                        finger_cu[k] = midi2finger(finger2midi(finger_cu[k],tune=tune,capo=capo), tune=tune, capo=capo)[-1]
                    #print('note {}: {}, thr={}'.format(i, finger_cu, root_thr))
                    if (int(finger_cu[0][1]) <= 3) or (multi == 1) :  #当音簇最高音在1或2弦，或者音簇为多音时
                        #melody = finger2midi(finger_cu[0])  
                        indexlist[i] += 10
                        melody = str2midi(pitch_cu[0])
                        #print('    旋律音:{}, 其midi值为{}, 在{}'.format(midi2str(int(melody)), melody, finger_cu[0]))
                    if (int(finger_cu[-1][1]) >= root_thr):  #当音簇最低音在5或6弦时
                        root = str2midi(pitch_cu[-1]) 
                        indexlist[i] += 5
                        #print('    根音:{}, 其midi值{}, 在{}'.format(midi2str(int(root)), root, finger_cu[-1]))
                    #print('---------------')
                # roots.append(midi2str(int(root)))      
                # melodys.append(midi2str(int(melody)))
            #return [melodys, roots]  #返回该小节内所有满足条件的根音和旋律音

            for i in range(pitchnum):
                if indexlist[i] == 15:#当根音旋律音都有
                    pitchs[i] = pitchs[i].split()[0] + ' '+pitchs[i].split()[-1]
                    fingers[i] = fingers[i].split()[0] + ' '+fingers[i].split()[-1]
                elif indexlist[i] == 10:#只有旋律音
                    pitchs[i] = pitchs[i].split()[0]
                elif indexlist[i] == 5:#只有根音
                    fingers[i] = fingers[i].split()[-1]
            if indexlist[0] == 0:
                pitchs[0], fingers[0] = 'R', '(R,R)'
            for i in range(len(times)-1, -1, -1):
                if indexlist[i] == 0 and i >0:#当二者都没有时：
                    times[i-1] += times[i]
                    times[i] = 0 #将该时值交给前面
                    pitchs[i], fingers[i] = '', ''
            #最后清除空元素
            if clear_zero:
                for i in range(len(times)-1, -1, -1):
                    if times[i] == 0:
                        pitchs.pop(i)
                        fingers.pop(i)
                        times.pop(i)

            return1 = Tablature(pitch=pitchs, finger=fingers, time=times, info=self.info)
            readymeasures.append(return1)
            
        totalreturn = measure_join(readymeasures)  
        return totalreturn

    def writeTAB(self, filepath, divisions=16):   
        def make_notes(domtree, pitch_cu, timesig, finger_cu, divisions=16):

            '''
            针对的是音簇，即'4G 3G 3C'
            '''
            pitch_singles = pitch_cu.split()
            if_multi = 0
            if len(pitch_singles) > 1:
                if_multi = 1
            noteNodelist = []
            for i, single in enumerate(pitch_singles):

                if single != 'R':
                    octave, step = domtree.createTextNode(single[0]), domtree.createTextNode(single[1])
                    if len(single) > 2 and single[-1] == '#': 
                        alt=1
                    elif len(single) > 2 and single[-1] == '-':
                        alt=-1
                    else:
                        alt=0
                    #创建step和octave节点，并把具体值填入    
                    stepNode = domtree.createElement('step')
                    stepNode.appendChild(step)
                    octaveNode = domtree.createElement('octave')
                    octaveNode.appendChild(octave)
                    pitchNode = domtree.createElement('pitch')
                    alterNode = domtree.createElement('alter')

                    #将setp和octave节点添加到pitch节点,以及alter（如果有的话）
                    pitchNode.appendChild(stepNode)
                    if alt != 0:
                        alter = domtree.createTextNode(str(alt))
                        alterNode.appendChild(alter)
                        pitchNode.appendChild(alterNode)
                    pitchNode.appendChild(octaveNode)  
                    noteNode = domtree.createElement('note')
                    if i >= 1:
                        chord = domtree.createElement('chord')
                        noteNode.appendChild(chord) #如果是多音的话 给note再添加一个chord空节点                 
                    noteNode.appendChild(pitchNode)
                elif single == 'R':
                    noteNode = domtree.createElement('note')
                    restNode = domtree.createElement('rest')
                    noteNode.appendChild(restNode)

                noteNodelist.append(noteNode)       

            '''
            为各个noteNode加上timesig与其他乱七八糟的节点
            noteNodelist针对的是音簇，即'4G 3G 3C'
            timesig是单独的时间，即'0.625'或0.0625.
            fingercu是指法簇，即'(1,3) (3,0) (5,3)'
            '''
            #将一个长的duration分解为不同的音符连起来
            res = []
            num = timesig / 4
            while num >= 1:
                res.append(4)
                num -= 1
            while num >= 0.5:
                res.append(2)
                num -= 0.5
            while num >= 0.25:
                res.append(1)
                num -= 0.25
            while num >= 0.125:
                res.append(0.5)
                num -= 0.125
            while num >= 0.0625:
                res.append(0.125)
                num -= 0.0625
            #print(res)

            fingerlist = finger_cu.split()
            #print('noteNodelist长度:{}'.format(len(noteNodelist)))
            for i, noteNode in enumerate(noteNodelist):         
                domtree = minidom.Document()
                #根据曲目的division计算每个音符的duration       
                duration = domtree.createTextNode(str(int(res[0] * divisions)))      
                #print('duration: {}'.format(duration))
                durationNode = domtree.createElement('duration')
                durationNode.appendChild(duration)   
                noteNode.appendChild(durationNode)

                #剩下就是添加一些有的没的
                #添加voice
                voiceNode = domtree.createElement('voice')
                voiceNode.appendChild(domtree.createTextNode(str(1)))
                noteNode.appendChild(voiceNode)


                #添加type
                typeNode = domtree.createElement('type')
                typedict = {
                    'whole':4,
                    'half':2,
                    'quarter':1,
                    'eighth':0.5,  
                    '16th':0.25,
                    '32nd':0.125
                }


                get_typename = lambda type1: [key for key, value in typedict.items() if value == type1]
                type1 = get_typename(res[0])
                #print('这个音符的type是 {}'.format(type1))
                typeNode.appendChild(domtree.createTextNode(type1[0]))
                noteNode.appendChild(typeNode)

                #添加stem（符干）
                stemNode = domtree.createElement('stem')
                stemNode.appendChild(domtree.createTextNode('up'))
                noteNode.appendChild(stemNode)

                #添加staff（谱）
                staffNode = domtree.createElement('staff')
                staffNode.appendChild(domtree.createTextNode('1'))
                noteNode.appendChild(staffNode)

                #添加notehead
                noteheadNode = domtree.createElement('notehead')
                noteheadNode.appendChild(domtree.createTextNode('normal'))
                noteNode.appendChild(noteheadNode)

                #添加notation和里面的technical
                if len(noteNode.getElementsByTagName('rest')) == 0:
                    notationNode = domtree.createElement('notation')
                    dynamicsNode = domtree.createElement('dynamics')
                    mfNode = domtree.createElement('mf')
                    dynamicsNode.appendChild(mfNode)
                    notationNode.appendChild(dynamicsNode)

                    string, fret = fingerlist[i].split(',')
                    string, fret = int(string[1:]), int(fret[:-1])

                    technicalNode = domtree.createElement('technical')
                    #GP7_processing = domtree.createProcessingInstruction('GP7', '<root><string>{}</string><fret>{}</fret></root>'.format(string, fret))

                    stringNode = domtree.createElement('string')
                    stringText = domtree.createTextNode(str(string))
                    stringNode.appendChild(stringText)

                    fretNode = domtree.createElement('fret')
                    fretText = domtree.createTextNode(str(fret))
                    fretNode.appendChild(fretText)

                    technicalNode.appendChild(stringNode)
                    technicalNode.appendChild(fretNode)

                    notationNode.appendChild(technicalNode)
                    noteNode.appendChild(notationNode)

            tiednotenum = len(res)  
            if tiednotenum > 1: #当单独的timesig已不够 需要由连音线表示时：
                midNode, endNode = [], []
                for node in noteNodelist:
                    mid1 = node.cloneNode(True)
                    midNode.append(mid1)
                    end1 = node.cloneNode(True)
                    endNode.append(end1)
                for i in range(len(noteNodelist)):
                    tiestartNode = domtree.createElement('tie')
                    tiestartNode.setAttribute('type', 'start')
                    noteNodelist[i].getElementsByTagName('notation')[0].appendChild(tiestartNode)
                midtienum = tiednotenum - 2
                for i in range(midtienum):
                    for j in range(len(midNode)):
                        tiestartNode = domtree.createElement('tie')
                        tiestartNode.setAttribute('type', 'start') 
                        tiestopNode = domtree.createElement('tie')
                        tiestopNode.setAttribute('type', 'stop') 

                        extra_duration = res[i+1] * divisions
                        extra_type = get_typename(extra_duration)

                        midNode[j].getElementsByTagName('duration')[0].firstChild.data = str(int(extra_duration))
                        midNode[j].getElementsByTagName('type')[0].firstChild.data = extra_type
                        midNode[j].getElementsByTagName('notation')[0].appendChild(tiestartNode)
                        midNode[j].getElementsByTagName('notation')[0].appendChild(tiestopNode)
                    noteNodelist.extend(midNode)

                for i in range(len(endNode)):
                    tiestopNode = domtree.createElement('tie')
                    tiestopNode.setAttribute('type', 'stop')

                    extra_duration = res[-1] * divisions
                    extra_type = get_typename(extra_duration)
                    print(endNode[i].getElementsByTagName('type')[0].firstChild.data)
                    print('原duration：{}'.format(endNode[i].getElementsByTagName('duration')[0].firstChild.data))
                    endNode[i].getElementsByTagName('duration')[0].firstChild.data = str(int(extra_duration)) 
                    endNode[i].getElementsByTagName('type')[0].firstChild.data = extra_type
                    print('现duration:{}'.format(endNode[i].getElementsByTagName('duration')[0].firstChild.data))
                    endNode[i].getElementsByTagName('notation')[0].appendChild(tiestopNode)
                noteNodelist.extend(endNode)         
            return noteNodelist
        testnotes, testfingers, testtimes = self.pitch, self.finger, self.time
        #division采用16
        cwd = os.getcwd()
        domtree = parse(cwd + '\\head.xml')
        measurenum = len(testnotes)
        first_measure = domtree.getElementsByTagName('measure')[0]
        divisionsNode = first_measure.getElementsByTagName('divisions')
        divisionsNode[0].firstChild.data = divisions


        #创建一些空measure，并存放在list里备用
        measurelist = []
        for i in range(measurenum-1):
            measure = domtree.createElement('measure')
            measure.setAttribute('number', '{}'.format(i+2))
            measurelist.append(measure)

        for i in range(measurenum):  
            measurenote, measuretime, measurefinger = testnotes[i], testtimes[i], testfingers[i]
            all_noteNodes = []
            for j, note in enumerate(measurenote):
                #print('{} processing...'.format(j))
                noteNodelist = make_notes(domtree, note, timesig=measuretime[j], finger_cu=measurefinger[j], divisions=divisions)
                print(noteNodelist)
                #add_timesig(division=16, noteNodelist=noteNodelist, timesig=measuretime[j], finger_cu=measurefinger[j])
                #第一次循环中，将各音符Node添加到原有的首小节内
                if i <= 0:
                    for Node in noteNodelist:
                        first_measure.appendChild(Node)
                #其余的循环中，将各音符Node添加到空白的小节内
                else:
                    for Node in noteNodelist:
                        measurelist[i-1].appendChild(Node)

            all_measures = [first_measure]    
            all_measures.extend(measurelist)

            backupNodes = domtree.createElement('backup')
            duration_in_backupNodes = domtree.createElement('duration')
            all_timesig = divisions * 4
            _ = domtree.createTextNode(str(all_timesig))

            duration_in_backupNodes.appendChild(_)
            backupNodes.appendChild(duration_in_backupNodes)
            all_measures[i].appendChild(backupNodes)

        partNode = domtree.getElementsByTagName('part')  
        #print(partNode)
        for measure in measurelist:
            partNode[0].appendChild(measure) 

        with open(filepath, 'w', encoding='utf-8') as f:
            domtree.writexml(f, addindent=' ', newl='\n', encoding='utf-8')
        print('文件已保存为{}'.format(filepath))
        return measurelist
    def showall(self):
        print('pitch: {}'.format(self.pitch))
        print('finger: {}'.format(self.finger))
        print('time: {}'.format(self.time))
        
def readTAB(path, pitchtype='default', show_processing=False, show_tuning=False):
    '''
    Read a TAB in musicXML form, return the pitch, finger position and note beat
    ========
    pitchtype: change the text order in tokenized note. For example 'default' for 5C#, and 'music21' for C#5
    tune: adjust for special tuning. For example, DADGAD tuning is [-2, 0, 0, 0, -2, -2]
    '''
    def clipinfo(clipname):
        clipname = clipname[:-4]
        structure = clipname[-1]

        startbar = int(clipname[clipname.index('_')+1:clipname.index('^')])
        endbar = int(clipname[clipname.index('^')+1:-1])
        originals = clipname[clipname.index('[')+1:clipname.index(']')]
        name = clipname[clipname.index(']')+1:clipname.index('_')]
        infodict = {
            'originals': originals,
            'name': name,
            'startbar': startbar,
            'endbar': endbar     
        }
        return infodict
  
    def fullinfo(fullname):
        index = fullname.split(']')
        original = index[0][1:]
        name = index[-1]
        if len(index) == 1:
            original=None
        infodict = {
            'original': original,
            'name': name[:-4],  
        }
        return infodict
  
    def check_repeat(measurelist):
        '''
        检查每个小节的反复记号与房子，并记录下来以便展开为时间序列
        '''
        repeatstart = '1'
        repeatend = '2'
        repeatmark = list(np.zeros(len(measurelist)))
        housemark = list(np.zeros(len(measurelist)))
        for num, measure in enumerate(measurelist):
            mark = ''
            if haveNodes(measure, 'ending'):
                endingNode = measure.getElementsByTagName('ending')
                endingnumber = endingNode[0].getAttribute('number').split(', ')
                mark += ''.join(endingnumber)
                for Node in endingNode:
                    if Node.getAttribute('type') == 'start':
                        mark += 'a'
                    if Node.getAttribute('type') == 'stop':
                        mark += 'b'
                        mark += Node.getAttribute('times')

            housemark[num] = mark   

        for num, measure in enumerate(measurelist):
            mark = ''    
            if haveNodes(measure, 'repeat'):
                repeatNode = measure.getElementsByTagName('repeat')
                for Node in repeatNode:
                    if Node.getAttribute('direction') == 'forward':
                        mark += 'a'
                    if Node.getAttribute('direction') == 'backward':
                        mark += 'b'
                        mark += Node.getAttribute('times')

            repeatmark[num] = mark
        return repeatmark, housemark
        
    def repeater(pitchs, repeatmark, housemark):
        '''
        Repeat the Da capo measures and process the different ending by given repeatmark
        '''
        if len([i for i in repeatmark if i != '']) == 0 and len([i for i in housemark if i != '']) == 0:
            return pitchs
        else:      
            play = []
            abpair = []
            housepair = []
            for i in range(len(repeatmark)):
                if repeatmark[i] == 'a':
                    start = i
                if 'b' in str(repeatmark[i]):
                    end = i
                    times = str(repeatmark[i])[str(repeatmark[i]).index('b')+1:]
                    abpair.append((start, end+1, int(times)))

            for i in range(len(housemark)):
                if 'a' in str(housemark[i]):
                    charge = int(housemark[i][:housemark[i].index('a')])

                    start = i
                if 'b' in str(housemark[i]):
                    end = i

                    housepair.append((start, end, charge))
            ready_for_repeat = pitchs[:abpair[0][0]]

            play = []
            for j in range(len(abpair)):
                repeattime = abpair[j][2]
                repeatunit = pitchs[abpair[j][0]:abpair[j][1]]
                frag_unit = repeatunit[:housepair[j][0]-abpair[j][0]]

                if j < len(abpair)-1:
                    singleunit = pitchs[abpair[j][1]:abpair[j+1][0]]
                else:
                    singleunit = pitchs[abpair[j][1]:]

                charge = housepair[j][-1]
                for i in range(repeattime):
                    if str(i+1) in str(charge):
                        for unit in repeatunit:
                            play.append(unit)
                    elif str(i+1) not in str(charge):
                        for unit in frag_unit:
                            play.append(unit)
                for unit in singleunit:
                    play.append(unit)
            return play
    if path[-2:] != '\\':
        path += '\\'
    XML_files = os.listdir(path)
    XML_files = [file for file in XML_files if '.xml' in file]

    TAB = []
    TABobjects = []
    for file_num, file in enumerate(XML_files):  
        print('#####{}. {}'.format(file_num, file))
        infodict = fullinfo(file)
        _head = np.zeros((6, 1))
        
        test_tree = parse(path + file)
        test_element = test_tree.documentElement
        #***八度变化***
        octavechangeNodes = test_element.getElementsByTagName('octave-change')
        if len(octavechangeNodes) != 0:
            octavechange = int(octavechangeNodes[0].firstChild.data)
        else:
            octavechange = 0
            
        #***capo****
        capo = 0
        if haveNodes(test_element, 'capo'):
            capo = int(test_element.getElementsByTagName('capo')[0].firstChild.data)
            gl_capo = capo
            print('capo {}'.format(capo))
            
        #***记录调弦信息, 特殊调弦处理***#***
        staff_tuning = test_element.getElementsByTagName('staff-tuning')
        tunings = [0, 0, 0, 0, 0, 0]
        for Node in staff_tuning:
            linenum = Node.getAttribute('line')
            tuning_octave = Node.getElementsByTagName('tuning-octave')[0].firstChild.data
            tuning_step = Node.getElementsByTagName('tuning-step')[0].firstChild.data
            tuning_alter = '0'
            if haveNodes(Node, 'tuning-alter'):
                tuning_alter = Node.getElementsByTagName('tuning-alter')[0].firstChild.data
            #print(tuning_octave + tuning_step + tuning_alter)
            tunings[int(linenum) - 1] = (str2midi(tuning_octave + tuning_step) + int(tuning_alter))
           
        standard = [40, 45, 50, 55, 59, 64]
        current = [i[1] - i[0] for i in zip(standard, tunings)]
        gl_tune = [i+12*octavechange for i in current]
        if show_tuning:
            print('special tuning:{}, octave-change:{}'.format(gl_tune, octavechange))
                    
        measures = test_element.getElementsByTagName('measure') #获取所有小节的nodes
        beats = getAttr(measures[0], 'beats')  #拍号
        beat_type = getAttr(measures[0], 'beat-type')
        timesig = []
        pitchs = []
        timesigs = []
        fingers = []
    
        repeatmark, housemark = check_repeat(measures)
        
        for measure_num, measure in enumerate(measures):
            if show_processing == True:
                print('measure {} processing...'.format(measure_num))
            #******获得divisions******
            if haveNodes(measure, 'divisions'):
                divisions = measure.getElementsByTagName('divisions')[0].firstChild.data[0]
                divisions = int(divisions)
                
            parts = get_parts(measure) 
            partdicts = []
            for part_num, part in enumerate(parts):
                if show_processing == True:
                    print('  part {} processing...'.format(part_num))
                partdict = {
                  'dictpitch':0, 
                  'dictfinger':0, 
                  'dicttime':0
                }
                #__timesig, __pitch, __finger = tie_clean(part, divisions=divisions, beats=beats)
                partdict['dicttime'], partdict['dictpitch'], partdict['dictfinger'] = tie_clean(part, divisions=divisions, beats=beats)
                partdicts.append(partdict)
                
            _pitch, _finger, _total_times = part_integrate(partdicts)
            _total_times.append(int(beats))
            _timesig = list(np.diff(np.array(_total_times)))
            # print('timesig:{}'.format(timesig))
            # print('pitch:{}'.format(pitch))
            # print('finger:{}'.format(finger))
            should = float(beats) / (float(beat_type)//4)
            try:
                assert abs(sum(_timesig) - should) / should <= 0.05
            except:
                print('    ! In measure {}, sum of timesig should be {}, but{}'.format(measure_num, should, sum(_timesig)))
                                 
            assert len(_timesig)==len(_pitch)
            assert len(_timesig)==len(_finger)

            for i in range(len(_timesig)):
                _pitch[i] = str_postprocess(_pitch[i], pitchtype=pitchtype)
                _finger[i] = str_postprocess(_finger[i], mode='finger')
            timesigs.append(_timesig)
            pitchs.append(_pitch)
            fingers.append(_finger)
        
        #根据反复记号的mark来重复写一些小节
        #print('repeatmark: {}'.format(repeatmark))
        #print('housemark: {}'.format(housemark))
        pitchs = repeater(pitchs, repeatmark, housemark)
        fingers = repeater(fingers, repeatmark, housemark)
        timesigs = repeater(timesigs, repeatmark, housemark)
        #print(pitchs)     
        info = TABinfo(capo=capo, tuning=gl_tune, original=infodict['original'], name=infodict['name'], beat=beats, beat_type=beat_type)
        tab = Tablature(pitch=pitchs, finger=fingers, time=timesigs, info=info)
        
        TABobjects.append(tab)
    return TABobjects

def measure_join(measurelist):
    '''
    join many measure into one Tablature object
    '''
    pitchlist, timelist, fingerlist = [], [], []
    for measure in measurelist:
        pitchlist.append(measure.pitch)
        fingerlist.append(measure.finger)
        timelist.append(measure.time)
    
    return1 = Tablature(pitch=pitchlist, finger=fingerlist, time=timelist)
    return return1



# In[22]:


#关键变量：
#pitchset：list套list套list装字符串，包含音，如'4G 3G 3C'
#fingerset:list套list套list装字符串，包含弦与品，如（1，4）代表1弦4品
#timesigset：list套list套list装float，包含音对应的小节内时间，加和为1，如[0.5, 0.125, 0.125, 0.1875, 0.0625]


# In[23]:


def str2midi(str1):
    '''
    音簇字符串转换为midi值
    str1:字符串音高，例如4C，5D#，只能用升号，str
    如果只输入音名而不输入八度，例如G，则默认八度为5，即5G
    '''
    if not str1[0].isdigit():
        str1 = '5' + str1 
    midi_dict = {
        'C':0, 
        'D':2,
        'E':4,
        'F':5,
        'G':7,
        'A':9,
        'B':11
    }
    
    alt = 0
    if str1[-1] == '#':
        alt = 1
    if str1[-1] == '-':
        alt = -1
    
    if 'R' in str1:
        return None
    else:
        return 12 * (int(str1[0]) + 1)  + midi_dict['{}'.format(str1[1])] + alt

def midi2str(midi, alter='#'):
    '''
    midi值转为音簇字符串
    midi:midi值，int
    '''
    if midi == 0:
        return 'R'
    else:
        midilist = [0, 2, 4, 5, 7, 9, 11]
        octave = (midi // 12) - 1
        pitch = midi % 12
        if_alt = ''

        if pitch not in midilist:
            if alter == '#':
                if_alt = '#'
                pitch -= 1
            elif alter == '-':
                if_alt = '-'
                pitch += 1
        midi_dict = {
            'C':0, 
            'D':2,
            'E':4,
            'F':5,
            'G':7,
            'A':9,
            'B':11
        }
        pitchname = lambda midi: [key for key, value in midi_dict.items() if value == midi] #一行用值查找键 我太强叻
        str1 = str(octave) + pitchname(pitch)[0] + if_alt
        return str1

def finger2midi(finger, tune=[0, 0, 0, 0, 0, 0], capo=0):
    '''
    由吉他指法转换为midi值
    finger：指法，例如(1, 10)代表第一弦第十品，str格式
    tune为由低到高各弦的额外调音
    '''
    tune = [i + capo for i in tune]
    if finger == '(R,R)':
        return None
    else:
        string, fret = finger.split(',')
        string, fret = int(string[1:]), int(fret[:-1]) #得到第几弦和第几品
        standard = [64, 59, 55, 50, 45, 40]
        midi_of_string = [i[0] + i[1] for i in zip(standard, list(reversed(tune)))]
        return midi_of_string[string - 1] + fret #返回空弦midi值加品的midi值
    
def midi2finger(midi, tune=[0, 0, 0, 0, 0, 0], capo=0):
    '''
    通过给出的midi值推断指板位置。一个midi值可以对应多个位置，品味最高为24品
    '''
    string_dict = {
        '6':40,
        '5':45,
        '4':50,
        '3':55,
        '2':59,
        '1':64,
    }
    
    for i, tun in enumerate(list(reversed(tune))):
        string_dict[str(i+1)] += (tun+capo)

    fingers = []
    for string in string_dict.keys():
        if string_dict[string] <= midi and string_dict[string] + 24 >= midi:
            fret = midi - string_dict[string]
            string = int(string)
            fingers.append('({},{})'.format(string, fret))
    return fingers   


# In[24]:




#输出流水线制作xml


#用lettersong测试一下整体流水线，包括文件头制作、根音与旋律音识别、重新写入
def make_rootsong(filepath):
    _files = os.listdir(filepath)
    files = []
    for file in _files:
        if '.xml'  in file:
            files.append(file)
            
    miku_pitchset, miku_fingerset, miku_timeset = readTAB(filepath)
    melodyset, rootset = [], []
    for filenum, file in enumerate(files):
        print(file)
        headpath = make_xmlhead(filepath + file) #提取文件头
        index = list(zip(miku_pitchset[filenum], miku_fingerset[filenum], miku_timeset[filenum])) #第0首歌的音高、指法、时间戳
        print('#####小节数：{}#####'.format(len(miku_pitchset)))
        roots, melodys = [], []

        for measurenum, measure in enumerate(index):
            melody, root= root_detect(measure)
            print('#####{}小节结束#####'.format(measurenum))
            roots.append(root)
            melodys.append(melody)

        combineds = []    
        for i, measure_root in enumerate(zip(melodys, roots)):
            combined = []
            if i <= 2:
                print(measure_root)
            for i, j in zip(measure_root[0], measure_root[1]):
                cu = ' '
                if i != 'R':
                    cu += i
                    cu += ' '
                if j != 'R':
                    cu += j
                if cu == ' ':
                    cu = 'R'
                combined.append(cu)
            combineds.append(combined)
                       
        generateTAB(combineds, miku_fingerset[filenum], miku_timeset[filenum], filepath=headpath)
        melodyset.append(melodys)
        rootset.append(roots)
    
    return [melodyset, rootset]

def give_pitch(pitchset):
    for song in pitchset:
        measurenum = len(song)
        i = 0
        while i < measurenum:
            yield song[i]
            i+= 1

def melody_root_combine(melodys, roots):
    combineds = []    
    for i, measure_root in enumerate(zip(melodys, roots)):
        combined = []
        if i <= 2:
            print(measure_root)
        for i, j in zip(measure_root[0], measure_root[1]):
            cu = ' '
            if i != 'R':
                cu += i
                cu += ' '
            if j != 'R':
                cu += j
            if cu == ' ':
                cu = 'R'
                
            if cu[0] == ' ': 
                cu = cu[1:]            
            if cu[-1] == ' ':
                cu = cu[:-1]
            combined.append(cu)
            
        combineds.append(combined)
    return combineds
  
def split(li):
    return sum(([x] if not isinstance(x, list) else split(x) for x in li), [])
  

def positioning(num, pitchset):
    '''
    求所有曲目中的第num个音在第几首歌的第几小节
    '''
    count=0
    for songnum, song in enumerate(pitchset):
        for measurenum, measure in enumerate(song):
            for pitch_cu_num, pitch_cu in enumerate(measure):
                count += 1
                if count == num:
                    print('song{} measure{} note{}'.format(songnum, measurenum, pitch_cu_num))
                    
                    
#和弦识别
def chord_recognize(pitch_cu, return_overtone=False):
    '''
    大三和弦：有且仅有大三度和五度， 435 435 435 或 75 或 48 39
    小三和弦：有且仅有小三度和五度， 345 345 345
    大七和弦：存在大七度和大三度， 4341 4341 4341或471
    小七和弦：存在小七度和小三度， 3432 3432 3432或732
    属七和弦：存在小七度和大三度， 4332 4332 4332或732
    增三和弦：存在增五度（小六度）， 444 444 444
    减三和弦：存在增四度， 326 326 326或
    挂留2和弦：存在二度但不存在三度， 255 255 255
    挂留4和弦：存在四度但不存在三度， 525 525 525
    '''
    in_chord_pitchs = [i[1:] for i in pitch_cu.split()][-4:]
    pitchs = list(set(in_chord_pitchs))
    singles = list(np.sort([str2midi(i) for i in pitchs]))
    #print(singles)
    _ = singles
    singles.extend([i + 12 for i in _])
    singles.extend([i + 24 for i in _])
    
    delta_singles = []
    for i in range(len(singles)):
        if i > 0:
            delta_singles.append(singles[i] - singles[i-1])
    inverts = delta_singles
    feasure = ''
    for num in delta_singles:
        feasure += str(num)
    def ifin(str1):
        strlen = len(str1)
        newstrs = []
        for i in range(len(str1)):
            str1 = str1[1:] + str1[0]
            newstrs.append(str1)
        for newstr in newstrs:
            if newstr in feasure:
                return True
        return False
    #print(feasure)
    chord_type = 'NotChord'
    chordchar = 'NotChord'
    if ifin('255'): #当一阶差分只有5半音和2半音时 说明是挂2
        chord_type = 'sus'
        
        chordchar = midi2str(singles[feasure.index('2')])[1:]#2 前面的音符是挂2的根音

    elif ifin('336'):
        chord_type =  'dim'
        chordchar = midi2str(singles[feasure.index('6')+1])[1:]#6后面的是根音
    elif ifin('444'): #aug时返回最低的音
        chord_type = 'aug'
        chordchar = in_chord_pitchs[0]
    #七和弦组：
    
    if ifin('2') or ifin('1'): 
        #属七：
        if ifin('4332') or ifin('246'):
            chord_type = '7'
            chordchar = midi2str(singles[feasure.index('4')])[1:] #4前面
        #小七：              
        elif ifin('3432') or ifin('723') or ifin('732'):
            chord_type = 'm7'
            chordchar = midi2str(singles[feasure.index('2')+1])[1:] #2后面
        elif ifin('4341') or ifin('471') or ifin('741'):
            chord_type = 'maj7'
            chordchar = midi2str(singles[feasure.index('1')+1])[1:] #1后面
    #三和弦组：
    else:
        if ifin('345') or ifin('39'):
            chord_type = 'm'
            chordchar = midi2str(singles[feasure.index('3')])[1:] #3前面
        elif ifin('435') or ifin('75'):
            chord_type = 'maj'
            chordchar = midi2str(singles[feasure.index('5')+1])[1:] #5后面
        elif ifin('48') :
            chord_type = 'maj'
            chordchar = midi2str(singles[feasure.index('4')])[1:] #4前面
    if return_overtone == True:
        return [chord_type, singles]
    else:
        return chordchar, chord_type
    
    
  
def show_Nodes(Nodes, filename='show'):
    doc = minidom.Document()
    domtree =doc.createElement('domtree')
    for i, node in enumerate(Nodes):
        print(i)
        domtree.appendChild(node)
        
    print(domtree.childNodes)   
    path = os.getcwd() + '\\{}.xml'.format(filename)
    with open(path, 'w', encoding='utf-8') as f:
        domtree.writexml(f, addindent=' ', newl='\n')