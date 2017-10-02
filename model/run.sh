svm-scale -s range aggre_train > train_data.scale
svm-scale -r range aggre_test > test_data.scale
svm-train -s 3 train_data.scale model
svm-predict test_data.scale model pred
paste -d ',' sub pred > pre_sub
echo -e "tollgate_id,time_window,direction,volume\n$(cat pre_sub)" > sub.csv
rm pre_sub
