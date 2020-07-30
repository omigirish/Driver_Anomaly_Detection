import ibm_db as db2
did=input("Enter Driver_id to check for driving behaviour: ")
# try:
dsn="DATABASE=BLUDB;HOSTNAME=dashdb-txn-sbox-yp-dal09-04.services.dal.bluemix.net;PORT=50000;PROTOCOL=TCPIP;UID=jxm90623;PWD=g59rhr+62bs94jwx;"
uid="jxm90623"
pwd="g59rhr+62bs94jwx"
conn = db2.connect(dsn,"", "")
sql = "SELECT * FROM ANOMALY  where ID ='" + did +"'"
print(sql)
stmt = db2.exec_immediate(conn, sql)
tuple = db2.fetch_tuple(stmt)
print("These are the anomalous driving trends detected:")
print()
print("INDEX:")
print()
print("DT : The Change in throtle in 0.25s")
print("DS : The Change in Steering Angle in 0.25s")
print("DB : The Change in  Break applies in 0.25s ")
print("lane_invade : No of times the Vehicle cut driving Lanes")
print("Eye_blinks: The No of frames for which the drivers eye was closed during 10s")
print("DV: Speed Change in 0.25 s")
print("S: Speed during Capture")
print()
print()

while tuple != False:
    print ("DT:", tuple[0],end=" ")
    print ("DS:", tuple[1],end=" ")
    print ("DB: ", tuple[3])
    print ("Eye_blinks: ", tuple[5],end=" ")
    print ("lane_invades : ", tuple[4])
    print ("DV: ", tuple[6],end=" ")
    print ("S: ", tuple[2]," kmph")
    print()
    print()
    tuple = db2.fetch_tuple(stmt)
# except:
#     print("Failed to connect to Cloud database try again.....")



 