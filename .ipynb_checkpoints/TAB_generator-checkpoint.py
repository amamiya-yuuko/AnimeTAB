#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from xml.dom import minidom
from xml.dom.minidom import parse
import os


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



# In[19]:


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

def finger_detect(measure):
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
            fingers.append(midi2finger(str2midi(pitch), tune=gl_tune)[-1])
    return fingers

def tie_clean(measure, divisions=divisions, beats=4):
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
    finger_cus = finger_detect(measure)
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

def str_postprocess(pitchstr, mode='pitch', pitchtype='default'):
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
        pitchs = [[i, finger2midi(i, tune=gl_tune)] for i in pitchstr.split() if i != '(R,R)']
        newlist = sorted(pitchs, key = lambda x: x[1])
        for pitch in newlist:
            new_cu += (pitch[0] + ' ')
        if new_cu == '':
            return '(R,R)'
        else:
            return new_cu[:-1]

global gl_tune
gl_tune = [0, 0, 0, 0, 0, 0]
def readTAB(path, pitchtype='default', show_processing=False, show_tuning=False):
    '''
    Read a TAB in musicXML form, return the pitch, finger position and note beat
    ========
    pitchtype: change the text order in tokenized note. For example 'default' for 5C#, and 'music21' for C#5
    tune: adjust for special tuning. For example, DADGAD tuning is [-2, 0, 0, 0, -2, -2]
    '''
    XML_files = os.listdir(path)
    XML_files = [file for file in XML_files if '.xml' in file]

    TAB = []
    pitchset = []
    timesigset = []
    fingerset = []
    for file_num, file in enumerate(XML_files):  
        print('#####{}. {}'.format(file_num, file))
        _head = np.zeros((6, 1))
        
        test_tree = parse(path + file)
        test_element = test_tree.documentElement
        #***记录调弦信息, 特殊调弦处理***#***
        staff_tuning = test_element.getElementsByTagName('staff-tuning')
        tunings = []
        for Node in staff_tuning:
            tuning_octave = Node.getElementsByTagName('tuning-octave')[0].firstChild.data
            tuning_step = Node.getElementsByTagName('tuning-step')[0].firstChild.data
            tuning_alter = '0'
            if haveNodes(Node, 'tuning-alter'):
                tuning_alter = Node.getElementsByTagName('tuning-alter')[0].firstChild.data
            #print(tuning_octave + tuning_step + tuning_alter)
            tunings.append(str2midi(tuning_octave + tuning_step) + int(tuning_alter))
           
        standard = [40, 45, 50, 55, 59, 64]
        current = [i[1] - i[0] for i in zip(standard, tunings)]
        gl_tune = current
        if show_tuning:
            print(gl_tune)
        measures = test_element.getElementsByTagName('measure') #获取所有小节的nodes
        beats = getAttr(measures[0], 'beats')  #拍号
        beat_type = getAttr(measures[0], 'beat-type')
        timesig = []
        pitchs = []
        timesigs = []
        fingers = []
        
        
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
            #vector = vector_generator(measure)
            #_head = np.hstack((_head, vector))
            #print('measure {} done.'.format(measure_num))
        _head = np.delete(_head, 0, axis=1)
        pitchset.append(pitchs)
        timesigset.append(timesigs)
        fingerset.append(fingers)
        TAB.append(_head)
    return [pitchset, fingerset, timesigset]


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

def finger2midi(finger, tune=[0, 0, 0, 0, 0, 0]):
    '''
    由吉他指法转换为midi值
    finger：指法，例如(1, 10)代表第一弦第十品，str格式
    tune为由低到高各弦的额外调音
    '''
    if finger == '(R,R)':
        return None
    else:
        string, fret = finger.split(',')
        string, fret = int(string[1:]), int(fret[:-1]) #得到第几弦和第几品
        standard = [64, 59, 55, 50, 45, 40]
        midi_of_string = [i[0] + i[1] for i in zip(standard, list(reversed(tune)))]
        return midi_of_string[string - 1] + fret #返回空弦midi值加品的midi值
    
def midi2finger(midi, tune=[0, 0, 0, 0, 0, 0]):
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
        string_dict[str(i+1)] += tun

    fingers = []
    for string in string_dict.keys():
        if string_dict[string] <= midi and string_dict[string] + 24 >= midi:
            fret = midi - string_dict[string]
            string = int(string)
            fingers.append('({},{})'.format(string, fret))
    return fingers   


# In[24]:


def root_detect(measure, tune=[0, 0, 0, 0, 0, 0]):
    '''
    根音和旋律音检测
    measure:一个三个元素的list，分别是小节内的所有音、指法、时间
    '''
    pitchs, fingers, times = measure[0], measure[1], measure[2] #分别代表音高、指法、时间

    pitchnum = len(pitchs)
    #print('待检测的音为{}'.format(pitchs))
    if_multi = np.zeros(pitchnum)
    roots = [] 
    melodys = []
        
    for i in range(pitchnum):  #该小节内的第i个音
        multi = 0
        pitch_cu = pitchs[i].split()
        finger_cu = fingers[i].split()
        if len(pitch_cu) >= 2:
            if_multi[i] = 1 #检测是否为多音
            multi = 1
        
        melody, root = 0, 0
        if 'R' not in finger_cu[0]:
            for i in range(len(finger_cu)):
                finger_cu[i] = midi2finger(finger2midi(finger_cu[i]), tune=tune)[-1]
            if (int(finger_cu[0][1]) <= 3) or (multi == 1) :  #当音簇最高音在1或2弦，或者音簇为多音时
                #melody = finger2midi(finger_cu[0])  
                melody = str2midi(pitch_cu[0])
                #print('    旋律音:{}, 其midi值为{}, 在{}'.format(midi2str(int(melody)), melody, finger_cu[0]))
            if (int(finger_cu[-1][1]) >= 5):  #当音簇最低音在5或6弦时
                root = str2midi(pitch_cu[-1]) 
                #print('    根音:{}, 其midi值{}, 在{}'.format(midi2str(int(root)), root, finger_cu[-1]))
            #print('---------------')
        roots.append(midi2str(int(root)))
        
        melodys.append(midi2str(int(melody)))
    return [melodys, roots]  #返回该小节内所有满足条件的根音和旋律音
       

#输出流水线制作xml
def make_notes(domtree, pitch_cu, division, timesig, finger_cu):
    '''
    针对的是音簇，即'4G 3G 3C'
    '''
    pitch_singles = pitch_cu.split()
    if_multi = 0
    if len(pitch_singles) > 1:
        if_multi = 1
    noteNodelist = []
    for i, single in enumerate(pitch_singles):
        alt = 0
        if single != 'R':
            octave, step = domtree.createTextNode(single[0]), domtree.createTextNode(single[1])
            if len(single) > 2: 
                alt=1
            #创建step和octave节点，并把具体值填入    
            stepNode = domtree.createElement('step')
            stepNode.appendChild(step)
            octaveNode = domtree.createElement('octave')
            octaveNode.appendChild(octave)
            pitchNode = domtree.createElement('pitch')
            alterNode = domtree.createElement('alter')

            #将setp和octave节点添加到pitch节点,以及alter（如果有的话）
            pitchNode.appendChild(stepNode)
            if alt == 1:
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
    fingerlist = finger_cu.split()
    if_dot = 0
    
    timesig = float(timesig)
    
    
    for i, noteNode in enumerate(noteNodelist):         
        domtree = minidom.Document()
        #根据曲目的division计算每个音符的duration       
        duration = domtree.createTextNode(str(int(4 * timesig * division)))          
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
            'whole':1,
            'half':0.5,
            'quarter':0.25,
            'eighth':0.125,  
            '16th':0.0625
        }
        
        res = []
        num = timesig * 4
        while num >= 1:
            res.append(1)
            num -= 1
        while num >= 0.5:
            res.append(0.5)
            num -= 0.5
        while num >= 0.25:
            res.append(0.25)
            num -= 0.25
        while num >= 0.125:
            res.append(0.125)
            num -= 0.125
        while num >= 0.0625:
            res.append(0.0625)
            num -= 0.0625
        
        
        
        #检查附点
        typename = lambda type1: [key for key, value in typedict.items() if value == type1]
        if typename(timesig) == []: #当时值未出现在其中时
            if typename(timesig / 1.5) != []:
                if_dot = 1
            if typename(timesig / 1.5) == []:
                typedict.values()-timesig
                
        #print(timesig)
        
        type1 = typename(timesig)
        if if_dot == 1:  #如果有附点
            type1 = typename(timesig / 1.5) #暂时只能识别附点 附点音符是原音符的1.5倍时长 所以除以1.5
            dotNode = domtree.createElement('dot')
            noteNode.appendChild(dotNode)
        if len(type1) != 0:
            typeNode.appendChild(domtree.createTextNode(type1[0]))
        else:
            typeNode.appendChild(domtree.createTextNode('quarter'))
        noteNode.appendChild(typeNode)

        #添加stem（符干）
        stemNode = domtree.createElement('stem')
        stemNode.appendChild(domtree.createTextNode('up'))
        noteNode.appendChild(stemNode)
        
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
            GP7_processing = domtree.createProcessingInstruction('GP7', '<root><string>{}</string><fret>{}</fret></root>'.format(string, fret))
            technicalNode.appendChild(GP7_processing)
            notationNode.appendChild(technicalNode)
            noteNode.appendChild(notationNode)
            
    return noteNodelist 

        
        
        
def generateTAB(testnotes, testfingers, testtimes, filepath):   
    #division采用16
    domtree = parse(filepath)
    measurenum = len(testnotes)
    first_measure = domtree.getElementsByTagName('measure')[0]
    divisionsNode = first_measure.getElementsByTagName('divisions')
    divisionsNode[0].firstChild.data = 16
    
    
    #创建一些空measure，并存放在list里备用
    measurelist = []
    for i in range(measurenum-1):
        measure = domtree.createElement('measure')
        measure.setAttribute('number', '{}'.format(i+2))
        measurelist.append(measure)
    
    for i in range(measurenum):  
        measurenote, measuretime, measurefinger = testnotes[i], testtimes[i], testfingers[i]
        for j, note in enumerate(measurenote):
            noteNodelist = make_notes(domtree, note, division=16, timesig=measuretime[j], finger_cu=measurefinger[j])
            #add_timesig(division=16, noteNodelist=noteNodelist, timesig=measuretime[j], finger_cu=measurefinger[j])
            #第一次循环中，将各音符Node添加到原有的首小节内
            if i < 1:
                for Node in noteNodelist:
                    first_measure.appendChild(Node)
            #其余的循环中，将各音符Node添加到空白的小节内
            elif i >= 1:
                for Node in noteNodelist:
                    measurelist[i-1].appendChild(Node)
                        
    partNode = domtree.getElementsByTagName('part')  
    print(partNode)
    for measure in measurelist:
        partNode[0].appendChild(measure) 
        
    name = filepath.split('\\')[-1]
    print('文件名为：{}'.format(name))
    
    newfilepath = '\\'.join(filepath.split('\\')[:-1])
    with open(newfilepath + '\\new_{}'.format(name), 'w', encoding='utf-8') as f:
        domtree.writexml(f, addindent=' ', newl='\n', encoding='utf-8')
    print('文件已保存为{}\\new_{}'.format(newfilepath, name))





# In[28]:


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
  
def key_detect(pitchlist, return_vector=False):
    '''
    pitchlist: a list of measures, every measure is a list of clusters 
    '''
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

def key_shift(pitchlist, origin_key, new_key):
    '''
    给小节移调的
    '''
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
        
    return new_measure 

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
    pitchs = list(set([i[1:] for i in pitch_cu.split()][-4:]))
    singles = list(np.sort([str2midi(i) for i in pitchs]))
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
    
    if ifin('255'): #当一阶差分只有5半音和2半音时 说明是挂2或者挂4
        chord_type = 'sus'

    elif ifin('326'):
        chord_type =  'dim'
    elif ifin('444'):
        chord_type = 'aug'
    elif ifin('4332') or ifin('732'):
        chord_type = '7'
    elif ifin('3432'):
        chord_type = 'm7'
    elif ifin('4341') or ifin('471'):
        chord_type = 'maj7'
    elif ifin('345'):
        chord_type = 'm'
    elif ifin('435') or ifin('75') or ifin('48') or ifin('39'):
        chord_type = 'maj'
    else:
        # print('{} not a chord'.format(pitchs))
        # print(delta_singles)
        chord_type = 'NotChord'

    if return_overtone == True:
        return [chord_type, singles]
    else:
        return chord_type  