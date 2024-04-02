with SEARCH_CTE as (
select distinct SEARCH.UserID
        , SEARCH.EventID
        , SEARCH.Tag
from SEARCH
left join SEARCH_TAG on SEARCH.SearchID = SEARCH_TAG.SearchID
where SEARCH.Tag is not null
and SEARCH.Tag != ''
and SEARCH_TAG.Tag is not null
        and SEARCH.UserID not in (1,4,5,6,9,16,18,55,87)
union
select distinct SEARCH.UserID
, SEARCH.EventID
        , SEARCH_TAG.Tag
from SEARCH
left join SEARCH_TAG on SEARCH.SearchID = SEARCH_TAG.SearchID
where SEARCH.Tag is not null
and SEARCH.Tag != ''
        and SEARCH.UserID not in (1,4,5,6,9,16,18,55,87)
),
PHOTO_ATHLETE_CTE as (
select distinct EVENT.EventID
, PHOTO.PhotoID
, PHOTO_ATHLETE.Tag
    from EVENT
    join PHOTO on EVENT.EventID = PHOTO.EventID
    join PHOTO_ATHLETE on PHOTO.PhotoID = PHOTO_ATHLETE.PhotoID
    where EVENT.EventType = 'Mountain Bike'
),
PURCHASE_CTE as (
select max(PURCHASE.PurchaseID) as PurchaseID
, PURCHASE.UserID
        , PURCHASE.PhotoID
        , PHOTO.EventID
from PURCHASE
join PHOTO on PURCHASE.PhotoID = PHOTO.PhotoID
join PHOTO_ATHLETE on PHOTO.PhotoID = PHOTO_ATHLETE.PhotoID
    group by PURCHASE.UserID
        , PURCHASE.PhotoID
        , PHOTO.EventID
    )
select PhotoID
, count(distinct Tag) as 'Tag Searched'
    , count(distinct UserID) as 'Users Searching'
    , sum(Purchased) as 'Purchases'
    , sum(Purchased) / count(distinct UserID, Tag) as 'Purchase Rate'
from (select SEARCH_CTE.UserID
    , SEARCH_CTE.EventID
    , SEARCH_CTE.Tag
, PURCHASE_CTE.PurchaseID
    , PHOTO_ATHLETE_CTE.PhotoID
    , case when PURCHASE_CTE.PurchaseID is not null
then 1
        else 0
        end as Purchased
from SEARCH_CTE
join PHOTO_ATHLETE_CTE on SEARCH_CTE.EventID = PHOTO_ATHLETE_CTE.EventID
and SEARCH_CTE.Tag = PHOTO_ATHLETE_CTE.Tag
left join PURCHASE_CTE on SEARCH_CTE.UserID = PURCHASE_CTE.UserID
and SEARCH_CTE.EventID = PURCHASE_CTE.EventID
                        and PHOTO_ATHLETE_CTE.PhotoID = PURCHASE_CTE.PhotoID) a
group by PhotoID
order by sum(Purchased) / count(distinct UserID,Tag) desc;