python3 ../produce_validtrain.py 2016-10-11 2016-10-18 -m svr
svm-scale -s range train_data > train_data.scale
svm-scale -r range valid_data > valid_data.scale
svm-train -s 3 train_data.scale model
svm-predict valid_data.scale model valid
cut -d ' ' -f 1 valid_data > valid_ans
echo "SVR"
python3 ../MAPE.py -m svr
python3 ../produce_traindata.py -m svr -s 2016-11-18
python3 ../produce_testdata.py -m svr -s 2016-11-18
svm-scale -s range train_data > train_data.scale
svm-scale -r range test_data > test_data.scale
svm-train -s 3 train_data.scale model

svm-predict train_data.scale model pred_train
cut -d ' ' -f 1 train_data > train_ans

svm-predict test_data.scale model pred
paste -d ',' sub pred > pre_sub
echo -e "tollgate_id,time_window,direction,volume\n$(cat pre_sub)" > sub.csv
rm pre_sub
