from bs4 import BeautifulSoup

xml_path = "D:/cancer/gan/lung/LIDC_dataset/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-30178/3000566.000000-03192/069.xml"
with open(xml_path, 'r') as xml_file:
        markup = xml_file.read()
xml = BeautifulSoup(markup, features="xml")
patient_id = xml.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text
reading_sessions = xml.LidcReadMessage.find_all("readingSession")
for reading_session in reading_sessions:
        nodules = reading_session.find_all("unblindedReadNodule")
        for nodule in nodules:
            nodule_id = nodule.noduleID.text
            print(nodule_id)
            rois = nodule.find_all("roi")
            print(rois)