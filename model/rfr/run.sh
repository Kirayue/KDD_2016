python3 ../produce_validtrain.py 2016-10-11 2016-10-17 -m rfr
python3 train_valid_rfr.py
echo "Random Forest Regression:"
python3 ../MAPE.py -m rfr
python3 ../produce_traindata.py -m rfr -s 2016-10-18
python3 ../produce_testdata.py -m rfr -s 2016-10-18
python3 train_test_rfr.py
paste -d ',' sub pred > pre_sub
echo -e "tollgate_id,time_window,direction,volume\n$(cat pre_sub)" > sub.csv
rm pre_sub
