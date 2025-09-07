"""
https://github.com/usgpo/bill-status/tree/main
https://www.congress.gov/help/legislative-glossary

test

PosixPath('/Users/galtay/repos/legisplain/congress-bulk/data/108/bills/hconres/hconres393/fdsys_billstatus.xml')
PosixPath('/Users/galtay/repos/legisplain/congress-bulk/data/115/bills/hr/hr3354/fdsys_billstatus.xml')
"""

from collections import Counter
import datetime
import json
from pathlib import Path
from typing import Literal, Self
import xml.etree.ElementTree as ET

from pydantic import BaseModel
import rich
from tqdm import tqdm
import xmltodict


def count_tags(xmls: list[str]) -> Counter:
    tags = Counter()
    for xml in xmls:
        bill = ET.fromstring(xml).find("bill")
        for xel in bill:
            tags[xel.tag] += 1
    return tags


def assert_has_one_key(d: dict, name: str | None = None) -> str:
    keys = list(d.keys())
    assert len(keys) == 1
    if name is not None:
        assert keys == [name]
    return keys[0]


def camel_case_to_snake_case(name: str) -> str:
    return "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")


def get_version(bill_status_dict: dict) -> str:
    version = bill_status_dict.get("version")
    if version is None:
        bill_dict = bill_status_dict["bill"]
        version = bill_dict.get("version")
    if version is None:
        raise ValueError("no version found")

    assert version in ("1.0.0", "3.0.0")
    return version


def get_policy_area(bill_dict: dict) -> str | None:
    policy_area = bill_dict.get("policyArea")
    if policy_area is None:
        return None
    else:
        return policy_area["name"]


class Activity(BaseModel):
    name: str
    date: datetime.datetime | None

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        assert isinstance(in_dict, dict)
        return cls(
            name=in_dict.get("name"),
            date=in_dict.get("date"),
        )

    @classmethod
    def list_from_dict(cls, in_dict: dict | None) -> list[Self]:
        if in_dict is None:
            return []
        item_key = assert_has_one_key(in_dict, "item")
        items = in_dict[item_key]
        assert isinstance(items, list)
        return [cls.from_dict(item) for item in items]


class Subcommittee(BaseModel):
    system_code: str
    name: str
    activities: list[Activity]

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        assert isinstance(in_dict, dict)
        return cls(
            system_code=in_dict.get("systemCode"),
            name=in_dict.get("name"),
            activities=Activity.list_from_dict(in_dict.get("activities")),
        )

    @classmethod
    def list_from_dict(cls, in_dict: dict | None) -> list[Self]:
        if in_dict is None:
            return []
        item_key = assert_has_one_key(in_dict, "item")
        items = in_dict[item_key]
        assert isinstance(items, list)
        return [cls.from_dict(item) for item in items]


class Committee(BaseModel):
    system_code: str
    name: str
    chamber: str | None
    type: str | None
    subcommittees: list[Subcommittee]
    activities: list[Activity]

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        assert isinstance(in_dict, dict)
        return cls(
            system_code=in_dict.get("systemCode"),
            name=in_dict.get("name"),
            chamber=in_dict.get("chamber"),
            type=in_dict.get("type"),
            subcommittees=Subcommittee.list_from_dict(in_dict.get("subcommittees")),
            activities=Activity.list_from_dict(in_dict.get("activities")),
        )

    @classmethod
    def list_from_dict(cls, in_dict: dict | None) -> list[Self]:
        if in_dict is None:
            return []

        if "billCommittees" in in_dict:
            # v1.0.0 need to peel outter key
            assert_has_one_key(in_dict, "billCommittees")
            in_dict = in_dict["billCommittees"]
            if in_dict is None:
                return []

        item_key = assert_has_one_key(in_dict, "item")
        items = in_dict[item_key]
        assert isinstance(items, list)
        return [cls.from_dict(item) for item in items]


class CommitteeReport(BaseModel):
    citation: str

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        assert isinstance(in_dict, dict)
        return cls(citation=in_dict.get("citation"))

    @classmethod
    def list_from_dict(cls, in_dict: dict | None) -> list[Self]:
        if in_dict is None:
            return []
        assert_has_one_key(in_dict, "committeeReport")
        items = in_dict["committeeReport"]
        if isinstance(items, dict):
            assert_has_one_key(items, "citation")
            return [cls(citation=items["citation"])]
        assert isinstance(items, list)
        return [cls.from_dict(item) for item in items]


class SourceSystem(BaseModel):
    name: str
    code: int | None

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        return cls(
            name=in_dict.get("name"),
            code=in_dict.get("code"),
        )


class Action(BaseModel):
    action_date: datetime.date
    text: str | None
    type: str | None
    action_code: str | None
    source_system: SourceSystem | None
    committees: list[Committee]

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        return cls(
            action_date=in_dict.get("actionDate"),
            text=in_dict.get("text"),
            type=in_dict.get("type"),
            action_code=in_dict.get("action_code"),
            source_system=SourceSystem.from_dict(in_dict.get("sourceSystem")),
            committees=Committee.list_from_dict(in_dict.get("committees")),
        )

    @classmethod
    def list_from_dict(cls, in_dict: dict | None) -> list[Self]:
        if in_dict is None:
            return []
        # we can have actionTypeCounts and actionByCounts
        # assert_has_one_key(in_dict, "item")
        items = in_dict["item"]
        assert isinstance(items, list)
        return [cls.from_dict(item) for item in items]


class RelationshipDetail(BaseModel):
    type: str
    identified_by: str

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        assert isinstance(in_dict, dict)
        return cls(
            type=in_dict.get("type"),
            identified_by=in_dict.get("identifiedBy"),
        )

    @classmethod
    def list_from_dict(cls, in_dict: dict | None) -> list[Self]:
        if in_dict is None:
            return []
        assert isinstance(in_dict, dict)
        item_key = assert_has_one_key(in_dict, "item")
        items = in_dict[item_key]
        assert isinstance(items, list)
        return [cls.from_dict(item) for item in items]


class RelatedBill(BaseModel):
    title: str | None
    congress: int
    number: int
    type: str
    latest_action: Action | None
    relationship_details: list[RelationshipDetail]

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        assert isinstance(in_dict, dict)
        return cls(
            title=in_dict.get("title"),
            congress=in_dict.get("congress"),
            number=in_dict.get("number"),
            type=in_dict.get("type"),
            latest_action=Action.from_dict(in_dict.get("latestAction")),
            relationship_details=RelationshipDetail.list_from_dict(
                in_dict.get("relationshipDetails")
            ),
        )

    @classmethod
    def list_from_dict(cls, in_dict: dict | None) -> list[Self]:
        if in_dict is None:
            return []
        assert isinstance(in_dict, dict)
        item_key = assert_has_one_key(in_dict, "item")
        items = in_dict[item_key]
        assert isinstance(items, list)
        return [cls.from_dict(item) for item in items]


class Title(BaseModel):
    title_type: str
    title: str
    bill_text_version_name: str | None
    bill_text_version_code: str | None
    source_system: SourceSystem | None
    chamber_code: str | None
    chamber_name: str | None
    parent_title_type: str | None
    title_type_code: int | None
    update_date: datetime.datetime | None

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        assert isinstance(in_dict, dict)
        return cls(
            title_type=in_dict.get("titleType"),
            title=in_dict.get("title"),
            bill_text_version_name=in_dict.get("billTextVersionName"),
            bill_text_version_code=in_dict.get("billTextVersionCode"),
            source_system=SourceSystem.from_dict(in_dict.get("sourceSystem")),
            chamber_code=in_dict.get("chamberCode"),
            chamber_name=in_dict.get("chamberName"),
            parent_title_type=in_dict.get("parentTitleType"),
            title_type_code=in_dict.get("titleTypeCode"),
            update_date=in_dict.get("updateDate"),
        )

    @classmethod
    def list_from_dict(cls, in_dict: dict) -> list[Self]:
        if in_dict is None:
            return []
        assert isinstance(in_dict, dict)
        item_key = assert_has_one_key(in_dict, "item")
        items = in_dict[item_key]
        assert isinstance(items, list)
        return [cls.from_dict(item) for item in items]


class Identifiers(BaseModel):
    bioguide_id: str
    lis_id: str | None
    gpo_id: str | None

    @classmethod
    def from_dict(cls, in_dict: dict) -> Self | None:
        if in_dict is None:
            return None
        return cls(
            bioguide_id=in_dict.get("bioguideId"),
            lis_id=in_dict.get("lisID"),
            gpo_id=in_dict.get("gpoId"),
        )


class Sponsor(BaseModel):
    bioguide_id: str
    full_name: str
    first_name: str
    last_name: str
    party: str
    state: str
    identifiers: Identifiers | None
    middle_name: str | None
    district: str | None
    is_by_request: str | None
    by_request_type: str | None

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        assert_has_one_key(in_dict, "item")
        items = in_dict["item"]
        # dedupe dicts
        items = list(set([json.dumps(x) for x in items]))
        items = [json.loads(x) for x in items]

        if len(items) != 1:
            rich.print(f"multiple sponsors detected, choosing the first one. {items}")
        item = items[0]
        assert isinstance(item, dict)

        # some amendment sponsors have only a name like "Rules Committee"
        # skip these for now
        if list(item.keys()) == ["name"]:
            return None

        return cls(
            bioguide_id=item["bioguideId"],
            full_name=item["fullName"],
            first_name=item["firstName"],
            last_name=item["lastName"],
            party=item["party"],
            state=item["state"],
            identifiers=Identifiers.from_dict(item.get("identifiers")),
            middle_name=item.get("middleName"),
            district=item.get("district"),
            is_by_request=item.get("is_by_request"),
            by_request_type=item.get("by_request_type"),
        )


class Cosponsor(BaseModel):
    bioguide_id: str
    full_name: str
    first_name: str
    last_name: str
    party: str
    state: str
    sponsorship_date: datetime.date
    is_original_cosponsor: bool
    identifiers: Identifiers | None
    middle_name: str | None
    district: str | None
    sponsorship_withdrawn_date: datetime.datetime | None

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        assert isinstance(in_dict, dict)
        return cls(
            bioguide_id=in_dict["bioguideId"],
            full_name=in_dict["fullName"],
            first_name=in_dict["firstName"],
            last_name=in_dict["lastName"],
            party=in_dict["party"],
            state=in_dict["state"],
            sponsorship_date=in_dict["sponsorshipDate"],
            is_original_cosponsor=in_dict["isOriginalCosponsor"],
            identifiers=Identifiers.from_dict(in_dict.get("identifiers")),
            middle_name=in_dict.get("middleName"),
            district=in_dict.get("district"),
            sponsorship_withdrawn_date=in_dict.get("sponsorshipWithdrawnDate"),
        )

    @classmethod
    def list_from_dict(cls, in_dict: dict | None) -> list[Self]:
        if in_dict is None:
            return []
        item_key = assert_has_one_key(in_dict, "item")
        items = in_dict[item_key]
        assert isinstance(items, list)
        return [cls.from_dict(item) for item in items]


class CboCostEstimate(BaseModel):
    pub_date: datetime.datetime
    title: str
    url: str
    description: str | None

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        assert isinstance(in_dict, dict)
        return cls(
            pub_date=in_dict.get("pubDate"),
            title=in_dict.get("title"),
            url=in_dict.get("url"),
            description=in_dict.get("description"),
        )

    @classmethod
    def list_from_dict(cls, in_dict: dict | None) -> list[Self]:
        if in_dict is None:
            return []
        assert isinstance(in_dict, dict)
        item_key = assert_has_one_key(in_dict, "item")
        items = in_dict[item_key]
        assert isinstance(items, list)
        return [cls.from_dict(item) for item in items]


class Law(BaseModel):
    type: str
    number: str

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        assert isinstance(in_dict, dict)
        return cls(
            type=in_dict.get("type"),
            number=in_dict.get("number"),
        )

    @classmethod
    def list_from_dict(cls, in_dict: dict | None) -> list[Self]:
        if in_dict is None:
            return []
        assert isinstance(in_dict, dict)
        item_key = assert_has_one_key(in_dict, "item")
        items = in_dict[item_key]
        assert isinstance(items, list)
        return [cls.from_dict(item) for item in items]


class Link(BaseModel):
    name: str
    url: str

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        assert isinstance(in_dict, dict)
        return cls(
            name=in_dict.get("name"),
            url=in_dict.get("url"),
        )

    @classmethod
    def list_from_dict(cls, in_dict: dict | None) -> list[Self]:
        if in_dict is None:
            return []
        assert isinstance(in_dict, dict)
        item_key = assert_has_one_key(in_dict, "link")
        items = in_dict[item_key]
        assert isinstance(items, list)
        return [cls.from_dict(item) for item in items]


class Note(BaseModel):
    text: str
    links: list[Link]

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        assert isinstance(in_dict, dict)
        return cls(
            text=in_dict.get("text") or in_dict.get("cdata").get("text"),
            links=Link.list_from_dict(in_dict.get("links")),
        )

    @classmethod
    def list_from_dict(cls, in_dict: dict | None) -> list[Self]:
        if in_dict is None:
            return []
        assert isinstance(in_dict, dict)
        item_key = assert_has_one_key(in_dict, "item")
        items = in_dict[item_key]
        assert isinstance(items, list)
        return [cls.from_dict(item) for item in items]


class AmendedBill(BaseModel):
    congress: int
    type: str
    origin_chamber: str
    origin_chamber_code: str
    number: int
    title: str
    update_date_including_text: datetime.datetime | None

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        assert isinstance(in_dict, dict)
        return cls(
            congress=in_dict.get("congress"),
            type=in_dict.get("type"),
            origin_chamber=in_dict.get("originChamber"),
            origin_chamber_code=in_dict.get("originChamberCode"),
            number=in_dict.get("number"),
            title=in_dict.get("title"),
            update_date_including_text=in_dict.get("updateDateIncludingText"),
        )


class RecordedVote(BaseModel):
    roll_number: int
    chamber: str
    congress: int
    date: datetime.datetime
    session_number: int
    url: str | None

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        assert isinstance(in_dict, dict)
        return cls(
            roll_number=in_dict.get("rollNumber"),
            chamber=in_dict.get("chamber"),
            congress=in_dict.get("congress"),
            date=in_dict.get("date"),
            session_number=in_dict.get("sessionNumber"),
            url=in_dict.get("url"),
        )

    @classmethod
    def list_from_dict(cls, in_dict: dict | None) -> list[Self]:
        if in_dict is None:
            return []
        assert isinstance(in_dict, dict)
        item_key = assert_has_one_key(in_dict, "recordedVote")
        items = in_dict[item_key]
        assert isinstance(items, list)
        return [cls.from_dict(item) for item in items]


class ActionAmendment(BaseModel):
    action_date: datetime.date
    text: str | None
    action_time: str | None
    links: list[Link]
    type: str | None
    action_code: str | None
    source_system: SourceSystem | None
    recorded_votes: list[RecordedVote]

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        assert isinstance(in_dict, dict)

        # cases
        # {} no text key
        # {"text": None}
        # {"text": "actual text we want"}
        # {"cdata": {}}
        # {"cdata": {"text": None}}
        # {"cdata": {"text": "actual text we want"}}

        if "text" in in_dict:
            text = in_dict["text"]
        elif "cdata" in in_dict:
            text = in_dict["cdata"].get("text")
        else:
            text = None

        return cls(
            action_date=in_dict.get("actionDate"),
            text=text,
            action_time=in_dict.get("actionTime"),
            links=Link.list_from_dict(in_dict.get("links")),
            type=in_dict.get("type"),
            action_code=in_dict.get("actionCode"),
            source_system=SourceSystem.from_dict(in_dict.get("sourceSystem")),
            recorded_votes=RecordedVote.list_from_dict(in_dict.get("recordedVotes")),
        )

    @classmethod
    def list_from_dict(cls, in_dict: dict | None) -> list[Self]:
        if in_dict is None:
            return []
        assert isinstance(in_dict, dict)
        # can have a count key too
        # assert_has_one_key(in_dict, "actions")
        items = in_dict["actions"]["item"]
        assert isinstance(items, list)
        return [cls.from_dict(item) for item in items]


class Amendment(BaseModel):
    number: int
    congress: int
    type: str
    description: str | None
    purpose: str | None
    update_date: datetime.datetime
    latest_action: ActionAmendment | None
    sponsor: Sponsor | None
    submitted_date: datetime.datetime
    chamber: str
    links: list[Link]
    actions: list[ActionAmendment]
    amended_bill: AmendedBill

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        assert isinstance(in_dict, dict)

        # xml2dict turns multiple "number" tags at the same level into lists
        # see  congress-bulk/data/115/bills/hr/hr3354/fdsys_billstatus.xml
        if isinstance(in_dict.get("number"), list):
            # skip duplicates
            return None

        return cls(
            number=in_dict.get("number"),
            congress=in_dict.get("congress"),
            type=in_dict.get("type"),
            description=in_dict.get("description"),
            purpose=in_dict.get("purpose"),
            update_date=in_dict.get("updateDate"),
            latest_action=ActionAmendment.from_dict(in_dict.get("latestAction")),
            sponsor=Sponsor.from_dict(in_dict.get("sponsors")),
            submitted_date=in_dict.get("submittedDate"),
            chamber=in_dict.get("chamber"),
            links=Link.list_from_dict(in_dict.get("links")),
            actions=ActionAmendment.list_from_dict(in_dict.get("actions")),
            amended_bill=AmendedBill.from_dict(in_dict.get("amendedBill")),
        )

    @classmethod
    def list_from_dict(cls, in_dict: dict | None) -> list[Self]:
        if in_dict is None:
            return []
        assert isinstance(in_dict, dict)
        item_key = assert_has_one_key(in_dict, "amendment")
        items = in_dict[item_key]
        assert isinstance(items, list)
        return [
            cls.from_dict(item) for item in items if cls.from_dict(item) is not None
        ]


class TextVersionBillStatus(BaseModel):
    type: str
    date: datetime.datetime | None = None
    url: str | None = None

    @staticmethod
    def get_url_from_tv_formats(formats_dict: dict | None) -> str | None:
        if formats_dict is None:
            return None

        assert isinstance(formats_dict, dict)
        assert_has_one_key(formats_dict, "item")
        fmt_items = formats_dict["item"]
        assert isinstance(fmt_items, list)
        if len(fmt_items) == 0:
            return None

        urls = list(set([fmt["url"] for fmt in fmt_items]))
        assert len(urls) == 1
        return urls[0]

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        assert isinstance(in_dict, dict)
        out_dict = {
            "type": in_dict.get("type"),
            "date": in_dict.get("date"),
            "url": cls.get_url_from_tv_formats(in_dict.get("formats")),
        }
        if all([v is None for v in out_dict.values()]):
            return None
        else:
            return cls(**out_dict)

    @classmethod
    def list_from_dict(cls, in_dict: dict | None) -> list[Self]:
        if in_dict is None:
            return []
        item_key = assert_has_one_key(in_dict, "item")
        items = in_dict[item_key]
        assert isinstance(items, list)
        return [
            cls.from_dict(item) for item in items if cls.from_dict(item) is not None
        ]


class Summary(BaseModel):
    version_code: str
    action_date: datetime.date
    action_desc: str
    update_date: datetime.datetime
    text: str

    @classmethod
    def from_dict(cls, in_dict: dict | None) -> Self | None:
        if in_dict is None:
            return None
        assert isinstance(in_dict, dict)
        return cls(
            version_code=in_dict["versionCode"],
            action_date=in_dict["actionDate"],
            action_desc=in_dict["actionDesc"],
            update_date=in_dict["updateDate"],
            text=in_dict.get("text") or in_dict.get("cdata").get("text"),
        )

    @classmethod
    def list_from_dict(cls, in_dict: dict) -> list[Self]:
        if in_dict is None:
            return []
        assert isinstance(in_dict, dict)

        # v1.0.0 need to peel outter key
        if "billSummaries" in in_dict:
            item_key = assert_has_one_key(in_dict, "billSummaries")
            in_dict = in_dict[item_key]
            if in_dict is None:
                return []
            assert isinstance(in_dict, dict)
            item_key = "item"
        else:
            item_key = "summary"

        assert_has_one_key(in_dict, item_key)
        items = in_dict[item_key]
        assert isinstance(items, list)
        return [cls.from_dict(item) for item in items]


class DublinCoreBillStatus(BaseModel):
    dc_format: str
    dc_language: str
    dc_rights: str
    dc_contributor: str
    dc_description: str


def get_subjects_1p0p0(bill_dict: dict) -> list[str]:
    subjects = bill_dict.get("subjects")
    if subjects is None:
        return []

    assert_has_one_key(subjects, "billSubjects")
    subjects = subjects["billSubjects"]
    subjects = subjects["legislativeSubjects"]
    if subjects is None:
        return []
    subject_items = subjects["item"]
    assert isinstance(subject_items, list)
    out = []
    for item in subject_items:
        assert isinstance(item, dict)
        out.append(item["name"])
    return out


def get_subjects_3p0p0(bill_dict: dict) -> list[str]:
    subjects = bill_dict.get("subjects")
    if subjects is None:
        return []

    subjects = subjects["legislativeSubjects"]
    if subjects is None:
        return []
    subject_items = subjects["item"]
    assert isinstance(subject_items, list)
    out = []
    for item in subject_items:
        assert isinstance(item, dict)
        out.append(item["name"])
    return out


class BillStatus(BaseModel):
    version: Literal["1.0.0", "3.0.0"]
    number: int
    bill_type: Literal[
        "HCONRES", "HJRES", "HR", "HRES", "S", "SCONRES", "SJRES", "SRES"
    ]

    update_date: datetime.datetime
    origin_chamber: Literal["House", "Senate"]
    introduced_date: datetime.date
    congress: int
    title: str
    sponsor: Sponsor | None = None
    cosponsors: list[Cosponsor]
    origin_chamber_code: Literal["H", "S"] | None
    policy_area: str | None
    titles: list[Title]
    subjects: list[str]
    text_versions: list[TextVersionBillStatus]
    actions: list[Action]
    summaries: list[Summary]
    latest_action: Action
    committees: list[Committee]
    committee_reports: list[CommitteeReport]
    related_bills: list[RelatedBill]
    laws: list[Law]
    notes: list[Note]
    cbo_cost_estimates: list[CboCostEstimate]
    amendments: list[Amendment]

    # only in version 1.0.0
    create_date: datetime.datetime | None

    # only in version 3.0.0
    update_date_including_text: datetime.datetime | None

    @classmethod
    def from_xml_str(cls, xml: str) -> Self:
        base_dict = xmltodict.parse(
            xml, force_list=("item", "summary", "amendment", "link", "recordedVote")
        )
        assert_has_one_key(base_dict, "billStatus")
        bs_dict = base_dict["billStatus"]
        version = get_version(bs_dict)

        bill_dict = bs_dict["bill"]
        if version == "1.0.0":
            subjects = get_subjects_1p0p0(bill_dict)
            number = bill_dict["billNumber"]
            bill_type = bill_dict["billType"]
            create_date = bill_dict["createDate"]
            update_date_including_text = None
        elif version == "3.0.0":
            subjects = get_subjects_3p0p0(bill_dict)
            number = bill_dict["number"]
            bill_type = bill_dict["type"]
            update_date_including_text = bill_dict["updateDateIncludingText"]
            create_date = None

        return cls(
            version=version,
            number=number,
            bill_type=bill_type,
            create_date=create_date,
            update_date_including_text=update_date_including_text,
            update_date=bill_dict["updateDate"],
            origin_chamber=bill_dict["originChamber"],
            introduced_date=bill_dict["introducedDate"],
            congress=bill_dict["congress"],
            title=bill_dict["title"],
            origin_chamber_code=bill_dict.get("originChamberCode"),
            policy_area=get_policy_area(bill_dict),
            sponsor=Sponsor.from_dict(bill_dict.get("sponsors")),
            cosponsors=Cosponsor.list_from_dict(bill_dict.get("cosponsors")),
            titles=Title.list_from_dict(bill_dict.get("titles")),
            subjects=subjects,
            text_versions=TextVersionBillStatus.list_from_dict(
                bill_dict.get("textVersions")
            ),
            actions=Action.list_from_dict(bill_dict.get("actions")),
            summaries=Summary.list_from_dict(bill_dict.get("summaries")),
            latest_action=Action.from_dict(bill_dict.get("latestAction")),
            committees=Committee.list_from_dict(bill_dict.get("committees")),
            committee_reports=CommitteeReport.list_from_dict(
                bill_dict.get("committeeReports")
            ),
            related_bills=RelatedBill.list_from_dict(bill_dict.get("relatedBills")),
            laws=Law.list_from_dict(bill_dict.get("laws")),
            notes=Note.list_from_dict(bill_dict.get("notes")),
            cbo_cost_estimates=CboCostEstimate.list_from_dict(
                bill_dict.get("cboCostEstimates")
            ),
            amendments=Amendment.list_from_dict(bill_dict.get("amendments")),
        )


if __name__ == "__main__":
    congress_bulk_path = Path("/Users/galtay/repos/legisplain/congress-bulk")
    data_path = Path(congress_bulk_path) / "data"
    xml_paths = sorted(list(data_path.rglob("fdsys_billstatus.xml")))

    def add_test_case(congress_bulk_path, xml_path):
        xml = xml_path.read_text().strip()
        rel_path = xml_path.relative_to(congress_bulk_path / "data")
        trg_path = Path("../test-data") / rel_path
        trg_path.parent.mkdir(exist_ok=True, parents=True)
        with trg_path.open("w") as fp:
            fp.write(xml)

    COUNTERS = {
        "version": Counter(),
        "origin_chamber": Counter(),
        "origin_chamber_code": Counter(),
        "type": Counter(),
        "bill_type": Counter(),
        "congress": Counter(),
        "titles_keys": Counter(),
        "sponsors_keys": Counter(),
        "cosponsors_keys": Counter(),
        "subjects_keys": Counter(),
    }

    for xml_path in tqdm(xml_paths):
        xml = xml_path.read_text().strip()

        base_dict = xmltodict.parse(
            xml, force_list=("item", "summary", "amendment", "link", "recordedVote")
        )
        assert_has_one_key(base_dict, "billStatus")
        bs_dict = base_dict["billStatus"]
        version = get_version(bs_dict)
        bill_dict = bs_dict["bill"]

        bs = BillStatus.from_xml_str(xml)
