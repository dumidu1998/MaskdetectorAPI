import streamlit as st
import os 
import pandas as pd
import webbrowser


source_link = 'https://www.github.com/Nary-Vip'



#DB Management
import sqlite3 as sq
conn = sq.connect("usermanagement.db")
c = conn.cursor()    
       
    
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS idpass(username TEXT , password TEXT)')

def add_useridp(username , password):
	c.execute('INSERT INTO idpass(username , password) VALUES (?,?)' , (username , password))
	conn.commit()

def rem_user(username):
	c.execute('DELETE FROM idpass WHERE username = ?;' , (username,))
	conn.commit()


def login_useridp(username , password):
	c.execute('SELECT * FROM idpass WHERE username = ? AND password = ?' , (username , password))
	data = c.fetchall()
	return data

def check_already_exists(username):
	c.execute('SELECT * FROM idpass WHERE username = ?' , (username,))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM idpass')
	data = c.fetchall()
	return data


def main():
	""" Simple LOgin/Signup pgm"""
	st.title("Attendance System and Mask Detection")
	menu = ["Home" , "SignUp" , "Login" , "About US" , "Source Code"]
	comp_name = st.sidebar.text("NaryRam Recognition Softwares Ltd.")
	choice = st.sidebar.selectbox("Sections :" , menu)

	if choice == "Home":
		st.subheader("Home")
		st.text("""This program is used to automatically register the attendance
using face data of the attendees, it also recognise the presence of mask.""")
		st.info("Login to start the recognition program , if you dont have credentials , SignUp")

	
	elif choice == "About US":
		st.subheader("Project Information and Source files")
		st.text("""This project is developed by Naresh Kumar and Sriram of M.sc(AIML) which consists of 
Atteandance System and Mask Detection. For any references or queries , follow my 
GitHub(Nary-Vip).""")

	elif choice == "Login":
		st.subheader("Login to access and start the recognition model.")


		username = st.sidebar.text_input("UserName :")
		password = st.sidebar.text_input("Password :" , type = "password")
		

		if st.sidebar.checkbox("Login"):
			create_usertable()
			check = login_useridp(username , password)

			if check:
				st.sidebar.success("Logged In as {}".format(username))
				task = st.selectbox("Admin Tasks : " , ["None" , "Start the program" ,"Show the attendance" , "Show the admins" , "clear admin"])
				if task == "Start the program":
					st.balloons()
					choose_pgm = st.radio("Select the program : ", ('None' , 'Attendance System', 'Mask Detection'))
					if choose_pgm == "None":
						st.text("Choose the below any program to start.")

					elif choose_pgm == "Attendance System":
						st.text("The camcoder popup will be shown now and start recognising.")
						os.system("python3 face_3.py")
					    
					
					else:
						os.system("python3 camera.py")
						st.text("The camcoder popup will be shown now and start finding for the mask.")


				elif task == "None":
					st.info("To Logout , go back to home or untick the login.")

                
				elif task == "Show the attendance":
					upload_attendance = st.file_uploader("Choose a Attendance sheet")
					if upload_attendance is not None:
  						df = pd.read_csv(upload_attendance)
  						st.write(df)


				elif task == "Show the admins":
					st.subheader("The current admins are :")
					view_all = view_all_users()
					data_frame = pd.DataFrame(view_all , columns = ["username" , "password"])
					st.dataframe(data_frame)


				elif task == "clear admin":
					rm_user = st.text_input("Enter the admin username you wish to remove from the database :")
					master_code = st.text_input("Enter the master code :")

					chk1 = check_already_exists(rm_user)

					if st.button("Remove") and rm_user:
						if chk1:
							if rm_user == "Naresh":
								st.warning("Master account can't be deleted")

							elif master_code == "5678" and rm_user != "Naresh":
								rem_user(rm_user)
								st.success("{} is removed from the admin database".format(rm_user))

							else:
								st.warning("Invalid master code")
					
						else:
							st.warning("This username doesn't exists in our database")

					st.info("Master code is to verify the authurization.")
				


			else:
				st.sidebar.warning("Invalid Credentials")

			


	elif choice == "SignUp":
		st.subheader("Create a new account")
		new_user = st.text_input("Enter user name :")
		new_pwd = st.text_input("Enter the password :" , type = 'password')
		admin_code = st.text_input("Enter the admin code :" , type = "password")
		st.info("Admin code is to verify the authenticity of new user , Ask naresh for the code")

		if st.button("SignUp"):
			chk = check_already_exists(new_user)
			if chk:
				st.warning("This username already exists , choose another username")

			if not chk:
				if new_user != "" and new_pwd != "" and admin_code == "9090":
					
					create_usertable()
					add_useridp(new_user, new_pwd)

					st.success("Your account succesfully created")
					st.info("Go to Login page and give your new credentials")

				if admin_code != "9090":
					st.warning("Admin code is wrong")

				elif new_user == "" and new_pwd == "":
					st.warning("Give the credentials properly")


	elif choice == "Source Code":
		st.text("For the source code , head over to my GitHub")
		if st.button('My GitHub'):
  			webbrowser.open_new_tab(source_link)
			




if __name__ == "__main__":
	main()
