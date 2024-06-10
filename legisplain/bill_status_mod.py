"""
https://github.com/usgpo/bill-status/tree/main
https://www.congress.gov/help/legislative-glossary
"""

from __future__ import annotations
from collections import Counter
import datetime
import json
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

from pydantic import BaseModel


def get_text_or_none(xel: Optional[Element] = None) -> Optional[str]:
    return xel.text if xel is not None else None


class Activity(BaseModel):
    name: str
    date: datetime.datetime

    @classmethod
    def list_from_xel(cls, xel: Optional[Element]) -> list[Activity]:
        result = []
        if xel is None:
            return result
        assert xel.tag == "activities"
        for item in xel:
            if item.tag != "item":
                continue
            activity = cls(
                name=get_text_or_none(item.find("name")),
                date=get_text_or_none(item.find("date")),
            )
            result.append(activity)
        return result


class Subcommittee(BaseModel):
    system_code: str
    name: str
    activities: list[Activity]

    @classmethod
    def list_from_xel(cls, xel: Optional[Element]) -> list[Subcommittee]:
        result = []
        if xel is None:
            return result
        assert xel.tag == "subcommittees"
        for item in xel:
            if item.tag != "item":
                continue
            subcommittee = cls(
                system_code=get_text_or_none(item.find("systemCode")),
                name=get_text_or_none(item.find("name")),
                activities=Activity.list_from_xel(item.find("activities")),
            )
            result.append(subcommittee)
        return result


class Committee(BaseModel):
    system_code: str
    name: str
    chamber: Optional[str] = None
    type: Optional[str] = None
    subcommittees: list[Subcommittee] = []
    activities: list[Activity] = []

    @classmethod
    def list_from_xel(cls, xel: Optional[Element]) -> list[Committee]:
        result = []
        if xel is None:
            return result
        assert xel.tag == "committees"
        for item in xel:
            if item.tag != "item":
                continue
            committee = cls(
                system_code=get_text_or_none(item.find("systemCode")),
                name=get_text_or_none(item.find("name")),
                chamber=get_text_or_none(item.find("chamber")),
                type=get_text_or_none(item.find("type")),
                subcommittees=Subcommittee.list_from_xel(item.find("subcommittees")),
                activities=Activity.list_from_xel(item.find("activities")),
            )
            result.append(committee)
        return result


class CommitteeReport(BaseModel):
    citation: str

    @classmethod
    def list_from_xel(cls, xel: Optional[Element]) -> list[CommitteeReport]:
        result = []
        if xel is None:
            return result
        assert xel.tag == "committeeReports"
        for item in xel:
            if item.tag != "committeeReport":
                continue
            committee_report = cls(
                citation=get_text_or_none(item.find("citation")),
            )
            result.append(committee_report)
        return result


class SourceSystem(BaseModel):
    name: str
    code: Optional[str] = None

    @classmethod
    def from_xel(cls, xel: Optional[Element]) -> Optional[SourceSystem]:
        if xel is None:
            return None
        else:
            assert xel.tag == "sourceSystem"
            return cls(
                name=get_text_or_none(xel.find("name")),
                code=get_text_or_none(xel.find("code")),
            )


class Action(BaseModel):
    action_date: datetime.date
    text: str
    type: Optional[str] = None
    action_code: Optional[str] = None
    source_system: Optional[SourceSystem] = None
    committees: list[Committee] = []

    @classmethod
    def from_xel(cls, xel: Optional[Element]) -> Optional[Action]:
        if xel is None or len(xel.findall("*"))==0:
            return None
        action = cls(
            action_date=get_text_or_none(xel.find("actionDate")),
            text=get_text_or_none(xel.find("text")),
            type=get_text_or_none(xel.find("type")),
            action_code=get_text_or_none(xel.find("actionCode")),
            source_system=SourceSystem.from_xel(xel.find("sourceSystem")),
            committees=Committee.list_from_xel(xel.find("committees")),
        )
        return action

    @classmethod
    def list_from_xel(cls, xel: Optional[Element]) -> list[Action]:
        result = []
        if xel is None:
            return result
        assert xel.tag == "actions"
        for item in xel:
            if item.tag != "item":
                continue
            action = cls.from_xel(item)
            if action is not None:
                result.append(action)
        return result


class RelationshipDetail(BaseModel):
    type: str
    identified_by: str

    @classmethod
    def list_from_xel(cls, xel: Optional[Element]) -> list[RelationshipDetail]:
        result = []
        if xel is None:
            return result
        assert xel.tag == "relationshipDetails"
        for item in xel:
            if item.tag != "item":
                continue
            relationship_detail = cls(
                type=get_text_or_none(item.find("type")),
                identified_by=get_text_or_none(item.find("identifiedBy")),
            )
            result.append(relationship_detail)
        return result


class RelatedBill(BaseModel):
    title: Optional[str]
    congress: int
    number: int
    type: str
    latest_action: Optional[Action] = None
    relationship_details: list[RelationshipDetail]

    @classmethod
    def list_from_xel(cls, xel: Optional[Element]) -> list[RelatedBill]:
        result = []
        if xel is None:
            return result
        assert xel.tag == "relatedBills"
        for item in xel:
            if item.tag != "item":
                continue
            related_bill = cls(
                title=get_text_or_none(item.find("title")),
                congress=get_text_or_none(item.find("congress")),
                number=get_text_or_none(item.find("number")),
                type=get_text_or_none(item.find("type")),
                latest_action=Action.from_xel(item.find("latestAction")),
                relationship_details=RelationshipDetail.list_from_xel(
                    item.find("relationshipDetails")
                ),
            )
            result.append(related_bill)
        return result


class Title(BaseModel):
    title_type: str
    title: str
    chamber_code: Optional[str] = None
    chamber_name: Optional[str] = None
    bill_text_version_name: Optional[str] = None
    bill_text_version_code: Optional[str] = None

    @classmethod
    def list_from_xel(cls, xel: Element) -> list[Title]:
        result = []
        if xel is None:
            return result
        assert xel.tag == "titles"
        for item in xel:
            if item.tag != "item":
                continue
            title = cls(
                title_type=get_text_or_none(item.find("titleType")),
                title=get_text_or_none(item.find("title")),
                chamber_code=get_text_or_none(item.find("chamberCode")),
                chamber_name=get_text_or_none(item.find("chamberName")),
                bill_text_version_name=get_text_or_none(
                    item.find("billTextVersionName")
                ),
                bill_text_version_code=get_text_or_none(
                    item.find("billTextVersionCode")
                ),
            )
            result.append(title)
        return result


class Identifiers(BaseModel):
    bioguide_id: str
    lis_id: Optional[str] = None
    gpo_id: Optional[str] = None

    @classmethod
    def from_xel(cls, xel: Optional[Element]) -> Optional[Identifiers]:
        if xel is None:
            return None
        else:
            assert xel.tag == "identifiers"
            return cls(
                lis_id=get_text_or_none(xel.find("lisID")),
                bioguide_id=get_text_or_none(xel.find("bioguideId")),
                gpo_id=get_text_or_none(xel.find("gpoId")),
            )


class Sponsor(BaseModel):
    bioguide_id: str
    full_name: str
    first_name: str
    last_name: str
    party: str
    state: str
    identifiers: Optional[Identifiers] = None
    middle_name: Optional[str] = None
    district: Optional[str] = None
    is_by_request: Optional[str] = None

    @classmethod
    def list_from_xel(cls, xel: Element) -> list[Sponsor]:
        result = []
        if xel is None:
            return result
        assert xel.tag == "sponsors"
        for item in xel:
            if item.tag != "item":
                continue
            # some amendment sponsors have only a name like "Rules Committee"
            # skip these for now
            if [el.tag for el in item] == ["name"]:
                continue
            sponsor = cls(
                bioguide_id=get_text_or_none(item.find("bioguideId")),
                full_name=get_text_or_none(item.find("fullName")),
                first_name=get_text_or_none(item.find("firstName")),
                last_name=get_text_or_none(item.find("lastName")),
                party=get_text_or_none(item.find("party")),
                state=get_text_or_none(item.find("state")),
                middle_name=get_text_or_none(item.find("middleName")),
                district=get_text_or_none(item.find("district")),
                is_by_request=get_text_or_none(item.find("isByRequest")),
                identifiers=Identifiers.from_xel(item.find("identifiers")),
            )
            result.append(sponsor)
        return result


class Cosponsor(BaseModel):
    bioguide_id: str
    full_name: str
    first_name: str
    last_name: str
    party: str
    state: str
    middle_name: Optional[str] = None
    district: Optional[str] = None
    sponsorship_date: datetime.date
    is_original_cosponsor: bool

    @classmethod
    def list_from_xel(cls, xel: Element) -> list[Cosponsor]:
        result = []
        if xel is None:
            return result
        assert xel.tag == "cosponsors"
        for item in xel:
            if item.tag != "item":
                continue
            cosponsor = cls(
                bioguide_id=get_text_or_none(item.find("bioguideId")),
                full_name=get_text_or_none(item.find("fullName")),
                first_name=get_text_or_none(item.find("firstName")),
                last_name=get_text_or_none(item.find("lastName")),
                party=get_text_or_none(item.find("party")),
                state=get_text_or_none(item.find("state")),
                middle_name=get_text_or_none(item.find("middleName")),
                district=get_text_or_none(item.find("district")),
                sponsorship_date=get_text_or_none(item.find("sponsorshipDate")),
                is_original_cosponsor=get_text_or_none(
                    item.find("isOriginalCosponsor")
                ),
            )
            result.append(cosponsor)
        return result


class CboCostEstimate(BaseModel):
    pub_date: datetime.datetime
    title: str
    url: str
    description: Optional[str]

    @classmethod
    def list_from_xel(cls, xel: Element) -> list[CboCostEstimate]:
        result = []
        if xel is None:
            return result
        assert xel.tag == "cboCostEstimates"
        for item in xel:
            if item.tag != "item":
                continue
            cbo_cost_estimate = cls(
                pub_date=get_text_or_none(item.find("pubDate")),
                title=get_text_or_none(item.find("title")),
                url=get_text_or_none(item.find("url")),
                description=get_text_or_none(item.find("description")),
            )
            result.append(cbo_cost_estimate)
        return result


class Law(BaseModel):
    type: str
    number: str

    @classmethod
    def list_from_xel(cls, xel: Element) -> list[CboCostEstimate]:
        result = []
        if xel is None:
            return result
        assert xel.tag == "laws"
        for item in xel:
            if item.tag != "item":
                continue
            law = cls(
                type=get_text_or_none(item.find("type")),
                number=get_text_or_none(item.find("number")),
            )
            result.append(law)
        return result


class Link(BaseModel):
    name: str
    url: str

    @classmethod
    def list_from_xel(cls, xel: Element) -> list[Link]:
        result = []
        if xel is None:
            return result
        assert xel.tag == "links"
        for item in xel:
            if item.tag != "link":
                continue
            link = cls(
                name=get_text_or_none(item.find("name")),
                url=get_text_or_none(item.find("url")),
            )
            result.append(link)
        return result


class Note(BaseModel):
    text: str
    links: list[Link]

    @classmethod
    def list_from_xel(cls, xel: Element) -> list[Note]:
        result = []
        if xel is None:
            return result
        assert xel.tag == "notes"
        for item in xel:
            if item.tag != "item":
                continue
            note = cls(
                text=get_text_or_none(item.find("text")),
                links=Link.list_from_xel(item.find("links")),
            )
            result.append(note)
        return result


class AmendedBill(BaseModel):
    congress: int
    type: str
    origin_chamber: str
    origin_chamber_code: str
    number: int
    title: str
    update_date_including_text: Optional[datetime.datetime]

    @classmethod
    def from_xel(cls, xel: Element) -> Optional[AmendedBill]:
        if xel is None:
            return None
        else:
            assert xel.tag == "amendedBill"
            return cls(
                congress=xel.find("congress").text,
                type=get_text_or_none(xel.find("type")),
                origin_chamber=xel.find("originChamber").text,
                origin_chamber_code=get_text_or_none(xel.find("originChamberCode")),
                number=get_text_or_none(xel.find("number")),
                title=xel.find("title").text,
                update_date_including_text=get_text_or_none(
                    xel.find("updateDateIncludingText")
                ),
            )


class RecordedVote(BaseModel):
    roll_number: int
    chamber: str
    congress: int
    date: datetime.datetime
    session_number: int
    url: Optional[str]

    @classmethod
    def list_from_xel(cls, xel: Element) -> list[RecordedVote]:
        result = []
        if xel is None:
            return result
        assert xel.tag == "recordedVotes"
        for item in xel:
            if item.tag != "recordedVote":
                continue
            recorded_vote = cls(
                roll_number=get_text_or_none(item.find("rollNumber")),
                chamber=get_text_or_none(item.find("chamber")),
                congress=get_text_or_none(item.find("congress")),
                date=get_text_or_none(item.find("date")),
                session_number=get_text_or_none(item.find("sessionNumber")),
                url=get_text_or_none(item.find("url")),
            )
            result.append(recorded_vote)
        return result


class ActionAmendment(BaseModel):
    action_date: datetime.date
    text: Optional[str]
    action_time: Optional[str] = None
    links: list[Link]
    type: Optional[str] = None
    action_code: Optional[str] = None
    source_system: Optional[SourceSystem] = None
    recorded_votes: Optional[list[RecordedVote]] = []

    @classmethod
    def from_xel(cls, xel: Element) -> ActionAmendment:
        if xel is None:
            return None
        else:
            assert xel.tag == "latestAction"
            return cls(
                action_date=get_text_or_none(xel.find("actionDate")),
                text=get_text_or_none(xel.find("text")),
                action_time=get_text_or_none(xel.find("actionTime")),
                links=Link.list_from_xel(xel.find("links")),
            )

    @classmethod
    def list_from_xel(cls, xel_outer: Element) -> list[ActionAmendment]:
        result = []
        if xel_outer is None:
            return result
        assert xel_outer.tag == "actions"
        count = int(xel_outer.find("count").text)
        xel = xel_outer.find("actions")
        if xel is None:
            return result

        for item in xel:
            if item.tag != "item":
                continue
            action_amendment = cls(
                action_date=get_text_or_none(item.find("actionDate")),
                action_time=get_text_or_none(item.find("actionTime")),
                text=get_text_or_none(item.find("text")),
                type=get_text_or_none(item.find("type")),
                action_code=get_text_or_none(item.find("actionCode")),
                source_system=SourceSystem.from_xel(item.find("sourceSystem")),
                recorded_votes=RecordedVote.list_from_xel(item.find("recordedVotes")),
                links=Link.list_from_xel(item.find("links")),
            )
            result.append(action_amendment)
        return result


class Amendment(BaseModel):
    number: int
    congress: int
    type: str
    description: Optional[str]
    purpose: Optional[str]
    update_date: datetime.datetime
    latest_action: Optional[ActionAmendment]
    sponsors: list[Sponsor]
    submitted_date: datetime.datetime
    chamber: str
    links: list[Link]
    actions: list[ActionAmendment]

    @classmethod
    def list_from_xel(cls, xel: Element) -> list[Amendment]:
        result = []
        if xel is None:
            return result
        assert xel.tag == "amendments"
        for item in xel:
            if item.tag != "amendment":
                continue
            amendment = cls(
                number=get_text_or_none(item.find("number")),
                congress=get_text_or_none(item.find("congress")),
                type=get_text_or_none(item.find("type")),
                description=get_text_or_none(item.find("description")),
                purpose=get_text_or_none(item.find("purpose")),
                update_date=get_text_or_none(item.find("updateDate")),
                latest_action=ActionAmendment.from_xel(item.find("latestAction")),
                sponsors=Sponsor.list_from_xel(item.find("sponsors")),
                submitted_date=get_text_or_none(item.find("submittedDate")),
                chamber=get_text_or_none(item.find("chamber")),
                amended_bill=AmendedBill.from_xel(item.find("amendedBill")),
                links=Link.list_from_xel(item.find("links")),
                actions=ActionAmendment.list_from_xel(item.find("actions")),
            )
            result.append(amendment)
        return result


class TextVersionBillStatus(BaseModel):
    type: str
    date: Optional[datetime.datetime] = None
    url: Optional[str] = None

    @classmethod
    def list_from_xel(cls, xel: Optional[Element]) -> list[TextVersionBillStatus]:
        result = []
        if xel is None:
            return result
        assert xel.tag == "textVersions"
        for item in xel:
            if item.tag != "item":
                continue
            formats = Format.list_from_xel(item.find("formats"))
            urls = list(set([el.url for el in formats]))
            if len(urls) == 0:
                url = None
            elif len(urls) == 1:
                url = urls[0]
            elif len(urls) > 1:
                raise ValueError("len(urls)>1")

            tv_type = get_text_or_none(item.find("type"))
            tv_date = get_text_or_none(item.find("date"))

            if url is None and tv_type is None and tv_date is None:
                continue
            else:
                text_version = cls(
                    type=tv_type,
                    date=tv_date,
                    url=url,
                )
                result.append(text_version)
        return result


class Format(BaseModel):
    url: str

    @classmethod
    def list_from_xel(cls, xel: Optional[Element]) -> list[Format]:
        result = []
        if xel is None:
            return result
        assert xel.tag == "formats"
        for item in xel:
            if item.tag != "item":
                continue
            el = cls(url=get_text_or_none(item.find("url")))
            result.append(el)
        return result


class Summary(BaseModel):
    version_code: str
    action_date: datetime.date
    action_desc: str
    update_date: datetime.datetime
    text: str

    @classmethod
    def list_from_xel(cls, xel: Optional[Element]) -> list[Summary]:
        result = []
        if xel is None:
            return result
        assert xel.tag == "summaries"
        for item in xel:
            if item.tag != "summary":
                continue
            summary = cls(
                version_code=get_text_or_none(item.find("versionCode")),
                action_date=get_text_or_none(item.find("actionDate")),
                action_desc=get_text_or_none(item.find("actionDesc")),
                update_date=get_text_or_none(item.find("updateDate")),
                text=get_text_or_none(item.find("text")),
            )
            result.append(summary)
        return result


class DublinCoreBillStatus(BaseModel):
    dc_format: str
    dc_language: str
    dc_rights: str
    dc_contributor: str
    dc_description: str

    @classmethod
    def from_xel(cls, xel: Element) -> DublinCoreBillStatus:
        assert xel.tag == "dublinCore"
        ns = {"dc": "http://purl.org/dc/elements/1.1/"}
        return cls(
            dc_format=xel.find("dc:format", namespaces=ns).text,
            dc_language=xel.find("dc:language", namespaces=ns).text,
            dc_rights=xel.find("dc:rights", namespaces=ns).text,
            dc_contributor=xel.find("dc:contributor", namespaces=ns).text,
            dc_description=xel.find("dc:description", namespaces=ns).text,
        )


def get_number(bill: Element) -> int:
    assert bill.tag == "bill"
    number = bill.find("number")
    bill_number = bill.find("billNumber")

    if number is None and bill_number is None:
        raise ValueError()
    elif number is None and bill_number is not None:
        return int(bill_number.text)
    elif number is not None and bill_number is None:
        return int(number.text)
    else:
        if number.text != bill_number.text:
            raise ValueError()
        return int(number.text)


def get_type(bill: Element) -> str:
    assert bill.tag == "bill"
    btype = bill.find("type")
    bill_type = bill.find("billType")

    if btype is None and bill_type is None:
        raise ValueError()
    elif btype is None and bill_type is not None:
        return bill_type.text
    elif btype is not None and bill_type is None:
        return btype.text
    else:
        if btype.text != bill_type.text:
            raise ValueError()
        return btype.text


def get_legislative_subjects(xel: Optional[Element]) -> list[str]:
    result = []
    if xel is None:
        return result
    assert xel.tag == "legislativeSubjects"
    for item in xel:
        if item.tag != "item":
            continue
        result.append(get_text_or_none(item.find("name")))
    return result


def get_subjects(xel: Optional[Element]) -> list[str]:
    result = []
    if xel is None:
        return result
    assert xel.tag == "subjects"
    result = get_legislative_subjects(xel.find("legislativeSubjects"))
    return result


def get_policy_area(xel: Optional[Element]) -> Optional[str]:
    if xel is None:
        return None
    assert xel.tag == "policyArea"
    return get_text_or_none(xel.find("name"))


class BillStatus(BaseModel):
    version: Optional[str] = None
    number: int
    update_date: datetime.datetime
    update_date_including_text: Optional[datetime.datetime] = None
    origin_chamber: str
    origin_chamber_code: Optional[str] = None
    type: str
    introduced_date: datetime.date
    congress: int
    committees: list[Committee]
    committee_reports: list[CommitteeReport]
    related_bills: list[RelatedBill]
    actions: list[Action]
    sponsors: list[Sponsor]
    cosponsors: list[Cosponsor]
    laws: list[Law]
    notes: list[Note]
    cbo_cost_estimates: list[CboCostEstimate]
    policy_area: Optional[str] = None
    subjects: list[str]
    summaries: list[Summary] = []
    title: str
    titles: list[Title]
    amendments: list[Amendment]
    text_versions: list[TextVersionBillStatus]
    latest_action: Action
    dublin_core: DublinCoreBillStatus

    @classmethod
    def from_xml_str(cls, xml: str):

        root = ET.fromstring(xml)
        version = root.find("version")
        bill = root.find("bill")

        return cls(
            version=get_text_or_none(version),
            number=get_number(bill),
            update_date=bill.find("updateDate").text,
            update_date_including_text=get_text_or_none(
                bill.find("updateDateIncludingText")
            ),
            origin_chamber=bill.find("originChamber").text,
            origin_chamber_code=get_text_or_none(bill.find("originChamberCode")),
            type=get_type(bill),
            introduced_date=bill.find("introducedDate").text,
            congress=bill.find("congress").text,
            committees=Committee.list_from_xel(bill.find("committees")),
            committee_reports=CommitteeReport.list_from_xel(
                bill.find("committeeReports")
            ),
            related_bills=RelatedBill.list_from_xel(bill.find("relatedBills")),
            actions=Action.list_from_xel(bill.find("actions")),
            sponsors=Sponsor.list_from_xel(bill.find("sponsors")),
            cosponsors=Cosponsor.list_from_xel(bill.find("cosponsors")),
            laws=Law.list_from_xel(bill.find("laws")),
            notes=Note.list_from_xel(bill.find("notes")),
            cbo_cost_estimates=CboCostEstimate.list_from_xel(
                bill.find("cboCostEstimates")
            ),
            policy_area=get_policy_area(bill.find("policyArea")),
            subjects=get_subjects(bill.find("subjects")),
            summaries=Summary.list_from_xel(bill.find("summaries")),
            title=bill.find("title").text,
            titles=Title.list_from_xel(bill.find("titles")),
            amendments=Amendment.list_from_xel(bill.find("amendments")),
            text_versions=TextVersionBillStatus.list_from_xel(
                bill.find("textVersions")
            ),
            latest_action=Action.from_xel(bill.find("latestAction")),
            dublin_core=DublinCoreBillStatus.from_xel(root.find("dublinCore")),
        )


def count_tags(xmls: list[str]) -> Counter:
    tags = Counter()
    for xml in xmls:
        bill = ET.fromstring(xml).find("bill")
        for xel in bill:
            tags[xel.tag] += 1
    return tags


