
def list_duplicates(seq):
    from collections import defaultdict
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() if len(locs)>1)
    
def check_results(all_list=[]):
    for list_ in all_list:
        print('----------------------------------')
        print(get_number_of_elements(list_))
        
        for dup in sorted(list_duplicates(list_)):
            print(dup)

# check_results([seq_id_list, label_list, label_id_list, pos_x_list, pos_y_list])
list_with_diff = []
for n in range(1, len(seq_id_list)):
    list_with_diff.append(seq_id_list[n] - seq_id_list[n-1])
