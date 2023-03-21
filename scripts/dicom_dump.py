import xnatutils

PROJECT_ID = "20230303102515basicmutablecs8de5"

with xnatutils.connect(server="http://localhost:8080", user="admin", password="admin") as xlogin:
    xproject = xlogin.projects[PROJECT_ID]
    for xsubject in xproject.subjects.values():
        for xsession in xsubject.experiments.values():
            for xscan in xsession.scans.values():
                scan_dicom_headers = xlogin.get(
                    f"/REST/services/dicomdump?src=/archive/projects/{PROJECT_ID}/"
                    f"subjects/{xsubject.id}/experiments/{xsession.id}/scans/{xscan.id}"
                ).json()["ResultSet"]["Result"]
