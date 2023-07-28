import os

target_dir = 'hif_ours'
source_dir = '/home/wjp21/project1/src/HiFaceGAN/results/hf_ours_split_final'
# os.system("cd /home/wjp21/project1/eval/data/"+target_dir)
# os.system("mkdir 1 2 3 4 5 6")
for i in range(1,7):
    os.system("cp -r "+source_dir+"/"+str(i)+"/*"+" /home/wjp21/project1/eval/data/"+target_dir+"/"+str(i))
    folder_path = "/home/wjp21/project1/eval/data/"+target_dir+"/"+str(i)
    for filename in os.listdir(folder_path):
        old_file_path = os.path.join(folder_path,filename)
        new_filename = "copy_"+filename
        new_file_path = os.path.join(folder_path,new_filename)
        os.rename(old_file_path,new_file_path)
    os.system("cp -r "+source_dir+"/"+str(i)+"/*"+" /home/wjp21/project1/eval/data/"+target_dir+"/"+str(i))
    os.system("cp -r /home/wjp21/project1/eval/data/train_base_ours/"+str(i)+"/*"+" /home/wjp21/project1/eval/data/"+target_dir+"/"+str(i))