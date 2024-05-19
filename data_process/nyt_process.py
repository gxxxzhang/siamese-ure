import json


def relation2id(dataset_name, label_name):
    label_id = []
    with open('../data/%s/nyt_su_rel2id_relation_span.json' % dataset_name, 'r') as f:
        data = json.load(f)
    for i in range(len(label_name)):
        id = data[label_name[i]]
        label_id.append(id)
    return label_id


def Read_nyt_data(CATEGORY):
    sentence = []
    label_name = []
    interval = ' '
    count = 0
    with open('../data/nyt+fb/nyt_ori_relation_span_%s.json' % CATEGORY, 'r') as f:
        line = f.readline()
        data = json.loads(line)
        print(len(data))
    for i in range(len(data)):
        tokens = data[i]["sentence"]
        subj_start = data[i]['head']['e1_begin']
        subj_end = data[i]['head']['e1_end']
        obj_start = data[i]['tail']['e2_begin']
        obj_end = data[i]['tail']['e2_end']
        relation = data[i]["relation"]
        subj_type = data[i]["head_type"]
        obj_type = data[i]["tail_type"]

        if obj_type not in ent_type:
            ent_type.append(obj_type)
        if subj_type not in ent_type:
            ent_type.append(subj_type)

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

        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')
        tokens = interval.join(tokens)

        sentence.append(tokens)
        label_name.append(relation)
        count = count + 1
        print(count)
    return sentence, label_name


if __name__ == '__main__':
    ent_type = []
    data_arr = ['train','test']
    for data_item in data_arr:
        NYT_sentence = []
        NYT_label_name = []
        NYT_label_id = []
        NYT_sentence, NYT_label_name = Read_nyt_data(data_item)
        NYT_label_id = relation2id('nyt+fb', NYT_label_name)

        with open('../data/nyt+fb/'+data_item+'_sentence.json', 'w') as fw:
            json.dump(NYT_sentence, fw, indent=4)

        with open('../data/nyt+fb/'+data_item+'_label_id.json', 'w') as fw:
            json.dump(NYT_label_id, fw, indent=4)
    print(ent_type)
