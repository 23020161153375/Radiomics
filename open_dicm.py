import dicom
f = dicom.read_file('1-001.dcm')
# print(f)
print(f[0x0008,0x0018].value)