
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import passwords
import time
import bisect
from datetime import datetime,timedelta
#driver = webdriver.Firefox()
options = Options()
options.headless = True
driver = webdriver.Firefox(options=options,firefox_binary="/usr/bin/firefox-esr")
# find all available slots does not include 30/30
monthtable = {'januari': 1,
              'februari': 2,
              'maart': 3,
              'april': 4,
              'mei': 5,
              'juni': 6,
              'juli': 7,
              'augustus': 8,
              'september': 9,
              'oktober': 10,
              'november': 11,
              'december': 12}
[62,150,237,324,411,498,585]
px_sgm=[138,288,438,588,738,888]

def get_to_correcttable(definedtime):
    # find current table timespan
    temp = driver.find_element_by_class_name('fc-header-title').text
    temp = temp.rsplit(' ')
    tabletimespan_start = datetime(int(temp[-1]), monthtable[temp[-2]], int(temp[0])).date()
    tabletimespan_end = datetime(int(temp[-1]), monthtable[temp[-2]], int(temp[-3])).date()
    while tabletimespan_start>definedtime:
        driver.find_element_by_xpath('//*[@class="fc-button fc-button-prev fc-state-default fc-corner-left"]').click()
        time.sleep(5)
        temp = driver.find_element_by_class_name('fc-header-title').text
        temp = temp.rsplit(' ')
        tabletimespan_start = datetime(int(temp[-1]), monthtable[temp[-2]], int(temp[0])).date()
        tabletimespan_end = datetime(int(temp[-1]), monthtable[temp[-2]], int(temp[-3])).date()
    while tabletimespan_end<definedtime:
        driver.find_element_by_xpath('//*[@class="fc-button fc-button-next fc-state-default fc-corner-right"]').click()
        time.sleep(5)
        temp = driver.find_element_by_class_name('fc-header-title').text
        temp = temp.rsplit(' ')
        tabletimespan_start = datetime(int(temp[-1]), monthtable[temp[-2]], int(temp[0])).date()
        tabletimespan_end = datetime(int(temp[-1]), monthtable[temp[-2]], int(temp[-3])).date()
def get_to_correctdate(definedtime):
    temp = driver.find_element_by_class_name('fc-header-title').text
    temp = temp.rsplit(' ')
    tabletime = datetime(int(temp[-1]), monthtable[temp[-2]], int(temp[-3])).date()
    while tabletime>definedtime:
        driver.find_element_by_xpath('//span[@class="fc-button fc-button-prev fc-state-default fc-corner-left"]').click()
        time.sleep(5)
        tabletime = datetime(int(temp[-1]), monthtable[temp[-2]], int(temp[-3])).date()
        time.sleep(5)
    while tabletime<definedtime:
        driver.find_element_by_xpath('//span[@class="fc-button fc-button-next fc-state-default fc-corner-right"]').click()
        time.sleep(5)
        tabletime = datetime(int(temp[-1]), monthtable[temp[-2]], int(temp[-3])).date()
        time.sleep(5)
    return
def get_date_of_slot(element):
    #Selected element as input.
    temp_str = element.get_attribute("style")
    temp_start=temp_str.find('left:')
    temp_end=temp_str.find('px;',temp_start)
    left_pos=int(temp_str[temp_start+5:temp_end])
    index=bisect.bisect_right(px_sgm, left_pos)
    temp = driver.find_element_by_class_name('fc-header-title').text
    temp = temp.rsplit(' ')
    time.sleep(5)
    temp_starttime = element.text.split()[0].rsplit(':')
    if not(temp[2]==temp[-2]) and (int(temp[0])<10):
        tabletimespan_start = datetime(int(temp[-1]), monthtable[temp[-2]], int(temp[0]), int(temp_starttime[0]),int(temp_starttime[1]))
    else:
        tabletimespan_start = datetime(int(temp[-1]), monthtable[temp[1]], int(temp[0]),int(temp_starttime[0]),int(temp_starttime[1]))

    element_date=tabletimespan_start+ timedelta(days=index)
    return element_date

def go_to_booking():
    driver.get('https://dmsonline.extern.kuleuven.be/nl/bookings')
    time.sleep(2)
    select = Select(driver.find_element_by_xpath("//select[@id='groupId']"))
    time.sleep(5)
    select.select_by_visible_text('Swimming')
    time.sleep(5)
    select = Select(driver.find_element_by_xpath("//select[@id='iResourceID']"))
    select.select_by_visible_text('Swimming Pool Reservation')
    time.sleep(5)

def get_exist_booking():
    driver.get('https://dmsonline.extern.kuleuven.be/nl/bookings/view/current')
    time.sleep(10)
    booked_slots_element=driver.find_elements_by_xpath('//tr[@id]')
    booked_date=[]
    if len(booked_slots_element)>0:
        for slot in booked_slots_element:
            date_string=slot.find_elements_by_xpath('td')[2].text
            time.sleep(5)
            date_object = datetime.strptime(date_string, "%d-%m-%Y %H:%M")
            booked_date.append(date_object)
    return booked_date

def book_slot(available):
    driver.execute_script("arguments[0].scrollIntoView(true);", available)
    time.sleep(5)
    ActionChains(driver).move_to_element(available).click().perform()
    # make reservation
    time.sleep(5)
    reserve = driver.find_element_by_xpath(
        '//input[@id="addBookingButton"and  @type="submit" and @value="Reserveren" ]')
    driver.execute_script("arguments[0].scrollIntoView(true);", reserve)
    time.sleep(5)
    driver.execute_script("arguments[0].click();", reserve)
    driver.implicitly_wait(10)
    # cancels the pop up window
    cancel = driver.find_element_by_xpath(
        '//a[@class="btn" and not(@aria-hidden="true") and contains(text(),"Sluiten")]')
    driver.execute_script("arguments[0].click();", cancel)


def book_date(definedtime):
    booked_slots = get_exist_booking()
    booked_dates = []
    [booked_dates.append(slot.date()) for slot in booked_slots]
    go_to_booking()
    get_to_correcttable(definedtime)
    availables = driver.find_elements_by_xpath(
        '//*[@class="fc-event fc-event-vert fc-event-start fc-event-end" and not(contains(., "[30/30]"))]')
    time.sleep(10)
    if len(availables) > 0:
        availables_today = []
        for available in availables:
            element_date = get_date_of_slot(available)
            if element_date.date() == definedtime:
                availables_today.append(available)
        if len(availables_today) > 0:
            available_today = availables_today[-1]
            element_date = get_date_of_slot(available_today)
            if (not(element_date.date() in booked_dates)) and (element_date-datetime.today()>timedelta(hours=1)) and (element_date.hour>20): 
                book_slot(available_today)







def main():

    try:
        driver.get("https://dmsonline.extern.kuleuven.be/nl/bookings")
        email = driver.find_element_by_name('email')
        password = driver.find_element_by_name('password')
        email.send_keys(passwords.email)
        password.send_keys(passwords.password)
        driver.find_element_by_id("submit").click()
        time.sleep(20)
        go_to_booking()
        time.sleep(10)
        #driver.find_element_by_xpath('//*[@class="fc-button fc-button-agendaDay fc-state-default fc-corner-right"]').click()
        #time.sleep(5)

        book_date(datetime.today().date() )
        print('book for '+str(datetime.today().date()))
        book_date(datetime.today().date() + timedelta(days=1))
        print('book for ' + str(datetime.today().date()))
        driver.close()
    except Exception as e:
        driver.close()
        print(e)
if __name__ == "__main__":
    main()


# make reservation
#reserve = driver.find_element_by_xpath('//input[@id="addBookingButton"and  @type="submit" and @value="Reserveren" and not(@disabled="")]').click()
