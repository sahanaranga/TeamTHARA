import re
import csv



def create_dataset(labels_path, output_path, articles_folder):

    f_output = open(output_path, 'w')

    with open(labels_path, 'r') as f:

        writer = csv.writer(f_output)
        # writer.writerow(['text', 'label'])
        article_file_name = ""
        af = None

        for line in f:
            fields = line.split('\t')
            id_file = fields[0]
            label = fields[1]
            begin_offset = int(fields[2])
            end_offset = int(fields[3])
            file_name = articles_folder + "article" + id_file + ".txt"

            if(file_name != article_file_name):
                if(af != None):
                    af.close()
                article_file_name = articles_folder + "article" + id_file + ".txt"
                af = open(article_file_name, 'rb')

            af.seek(begin_offset)
            assert end_offset - begin_offset > 0
            text = af.read(
                end_offset - begin_offset).decode('utf-8', 'ignore').strip()
            text = re.sub('[^A-Za-z0-9!?()%-= \']+', '', text)

            writer.writerow([text, label])

    af.close()
    f.close()
    f_output.close()




## main ##

train_labels = "ptc_corpus/ptc_datasets/train-task-flc-tc.labels"
train_articles = "ptc_corpus/ptc_datasets/train-articles/"
train_dataset_path = "ptc_corpus/ptc_train_dataset.csv"

val_labels = "ptc_corpus/ptc_datasets/dev-task-flc-tc.labels"
val_articles = "ptc_corpus/ptc_datasets/dev-articles/"
val_dataset_path = "ptc_corpus/ptc_validation_dataset.csv"

create_dataset(train_labels, train_dataset_path, train_articles)
create_dataset(val_labels, val_dataset_path, val_articles)