import json


def realtion2id(dataset_name,label_name):
    label_id=[]
    with open('../data/%s/rel2id.json'% dataset_name,'r') as f:
        data = json.load(f)
    for i in range(len(label_name)):
        id=data[label_name[i]]
        label_id.append(id)
    return label_id


def Read_TACRED_data(CATEGORY):
    sentence = []
    label_name = []
    interval=' '
    count=0
    with open('../data/tacred/%s.json' % CATEGORY, 'r') as f:
        line=f.readline()
        data=json.loads(line)
        print(len(data))
    for i in range(len(data)):
        tokens=data[i]["token"]
        subj_start=data[i]["subj_start"]
        subj_end=data[i]["subj_end"]
        obj_start=data[i]["obj_start"]
        obj_end=data[i]["obj_end"]
        relation=data[i]["relation"]
        obj_type = data[i]["obj_type"]
        subj_type = data[i]["subj_type"]
        if obj_type not in ent_type:
            ent_type.append(obj_type)
        if subj_type not in ent_type:
            ent_type.append(subj_type)
        if 'no_relation' in relation:
            continue
            pass
        print('{}: sub:{} , obj:{}'.format(relation, subj_type, obj_type))
        if subj_start < obj_start:
            tokens.insert(subj_start, '<e1:{}>'.format(subj_type))
            tokens.insert(subj_end + 2, '</e1:{}>'.format(subj_type))
            tokens.insert(obj_start + 2, '<e2:{}>'.format(obj_type))
            tokens.insert(obj_end + 4, '</e2:{}>'.format(obj_type))
        if subj_start == obj_start:
            tokens.insert(subj_start, '<e1:{}>'.format(subj_type))
            tokens.insert(subj_end + 2, '</e1:{}>'.format(subj_type))
            tokens.insert(obj_start + 1, '<e2:{}>'.format(obj_type))
            tokens.insert(obj_end + 3, '</e2:{}>'.format(obj_type))
        if subj_start > obj_start:
            tokens.insert(obj_start, '<e2:{}>'.format(obj_type))
            tokens.insert(obj_end + 2, '</e2:{}>'.format(obj_type))
            tokens.insert(subj_start + 2, '<e1:{}>'.format(subj_type))
            tokens.insert(subj_end + 4, '</e1:{}>'.format(subj_type))
        tokens.insert(0,'[CLS]')
        tokens.append('[SEP]')
        tokens=interval.join(tokens)

        sentence.append(tokens)
        label_name.append(relation)
        count=count+1
    print(count)
    return sentence,label_name


if __name__ == '__main__':
    ent_type= []
    data_arr = ['train','test','dev']
    for data_item in data_arr:
        TACRED_sentence = []
        TACRED_label_name = []
        TACRED_label_id = []
        TACRED_sentence, TACRED_label_name = Read_TACRED_data(data_item)
        TACRED_label_id = realtion2id('tacred', TACRED_label_name)

        with open('../data/tacred/'+data_item+'_sentence.json', 'w') as fw:
            json.dump(TACRED_sentence, fw, indent=4)

        with open('../data/tacred/'+data_item+'_label_id.json', 'w') as fw:
            json.dump(TACRED_label_id, fw, indent=4)

    print(ent_type)