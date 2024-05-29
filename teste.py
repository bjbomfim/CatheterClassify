
# Remover ids do csv
import pandas as pd

def remover_linhas_csv():
    
    # Ler o arquivo CSV
    df = pd.read_csv('trainteste.csv')

    # Lista de IDs a serem removidos
    ids_a_remover = ["00024905_000",
    "00024980_000",
    "00025068_000",
    "00025071_000",
    "00025071_001",
    "00025071_002",
    "00025086_000",
    "00025124_001",
    "00025124_002",
    "00025124_003",
    "00025124_004",
    "00025124_005",
    "00025124_006",
    "00025131_001",
    "00025128_003",
    "00025128_004",
    "00025128_005",
    "00025128_006",
    "00025128_007",
    "00025128_008",
    "00025131_002",
    "00025132_000",
    "00025132_002",
    "00025132_003",
    "00025137_000",
    "00025137_001",
    "00025137_002",
    "00025137_003",
    "00025147_000",
    "00025153_000",
    "00025153_001",
    "00025160_000",
    "00025168_000",
    "00025168_001",
    "00025168_002",
    "00025168_003",
    "00025168_004",
    "00025168_005",
    "00025188_000",
    "00025189_000",
    "00025189_001",
    "00025189_008",
    "00025189_009",
    "00025189_013",
    "00025189_014",
    "00025189_015",
    "00025189_016",
    "00025189_017",
    "00025189_018",
    "00025189_019",
    "00025213_000",
    "00025214_000",
    "00025220_000",
    "00025220_001",
    "00025220_002",
    "00025220_003",
    "00025220_004",
    "00025220_005",
    "00025220_006",
    "00025226_000",
    "00025226_001",
    "00025228_004",
    "00025228_005",
    "00025228_006",
    "00025238_000",
    "00025238_001",
    "00025238_002",
    "00025238_003",
    "00025238_004",
    "00025246_000",
    "00025251_000",
    "00025252_000",
    "00025252_001",
    "00025252_002",
    "00025252_003",
    "00025252_004",
    "00025252_005",
    "00025252_006",
    "00025252_007",
    "00025252_009",
    "00025252_0010",
    "00025252_0010",
    "00025252_0010",
    "00025252_0010",
    "00025252_0010",
    "00025252_0010",
    "00025252_0010",
    "00025252_0010",
    "00025252_0010",
    "00025252_0020",
    "00025252_0021",
    "00025252_0022",
    "00025252_0023",
    "00025252_0024",
    "00025252_0025",
    "00025252_0026",
    "00025252_041",
    "00025252_042",
    "00025252_043",
    "00025252_044",
    "00025252_045",
    "00025252_046",
    "00025252_047",
    "00025252_048",
    "00025252_049",
    "00025252_050",
    "00025252_051",
    "00025252_052",
    "00025252_053",
    "00025252_054",
    "00025252_055",
    "00025252_056",
    "00025252_057",
    "00025252_058",
    "00025252_059",
    "00025252_060",
    "00025252_061",
    "00025252_062",
    "00025252_063",
    "00025252_064",
    "00025252_065",
    "00025252_066",
    "00025261_000",
    "00025261_001",
    "00025261_002",
    "00025261_003",
    "00025261_004",
    "00025261_005",
    "00025262_000",
    "00025262_001",
    "00025262_002",
    "00025262_003",
    "00025262_004",
    "00025266_000",
    "00025282_000",
    "00025289_000",
    "00025289_001",
    "00025293_000",
    "00025294_000",
    "00025294_001",
    "00025294_002",
    "00025295_005",
    "00025295_006",
    "00025295_007",
    "00025301_000",
    "00025301_001",
    "00025304_000",
    "00025304_001",
    "00025317_000",
    "00025317_001",
    "00025317_002",
    "00025317_003",
    "00025317_004",
    "00025342_004",
    "00025342_005",
    "00025342_006",
    "00025342_007",
    "00025342_008",
    "00025342_009",
    "00025343_000",
    "00025352_000",
    "00025366_001",
    "00025368_002",
    "00025368_003",
    "00025368_004",
    "00025368_005",
    "00025368_006",
    "00025368_007",
    "00025368_008",
    "00025368_027",
    "00025369_000",
    "00025369_001",
    "00025369_002",
    "00025369_003",
    "00025369_004",
    "00025381_003",
    "00025381_006",
    "00025381_007",
    "00025381_008",
    "00025381_009",
    "00025381_010",
    "00025382_000",
    "00025382_001",
    "00025382_002",
    "00025382_003",
    "00025382_004",
    "00025382_005",
    "00025387_000",
    "00025387_001",
    "00025391_000",
    "00025391_001",
    "00025391_002",
    "00025391_003",
    "00025391_004",
    "00025393_000",
    "00025393_001",
    "00025401_000",
    "00025416_000",
    "00025416_001",
    "00025419_000",
    "00025426_000",
    "00025426_001",
    "00025426_002",
    "00025426_003",
    "00025451_000",
    "00025451_001",
    "00025451_002",
    "00025456_001",
    "00025456_002",
    "00025456_008",
    "00025461_001",
    "00025461_002",
    "00025461_003",
    "00025473_004",
    "00025473_005",
    "00025473_006",
    "00025473_007",
    "00025482_000",
    "00025482_001",
    "00025483_000",
    "00025483_001",
    "00025489_000",
    "00025489_001",
    "00025489_002",
    "00025495_001",
    "00025495_002",
    "00025495_003",
    "00025495_004",
    "00025495_005",
    "00025505_000",
    "00025505_001",
    "00025505_002",
    "00025510_002",
    "00025511_000",
    "00025513_000",
    "00025513_003",
    "00025513_011",
    "00025517_000",
    "00025518_000",
    "00025518_001",
    "00025518_002",
    "00025527_000",
    "00025527_002",
    "00025527_003",
    "00025529_009",
    "00025529_010",
    "00025529_011",
    "00025529_012",
    "00025529_013",
    "00025529_014",
    "00025529_015",
    "00025529_016",
    "00025529_017",
    "00025529_018",
    "00025529_019",
    "00025529_020",
    "00025529_021",
    "00025534_002",
    "00025534_003",
    "00025534_004",
    "00025534_005",
    "00025534_006",
    "00025537_002",
    "00025542_002",
    "00025542_003",
    "00025543_000",
    "00025543_001",
    "00025543_002",
    "00025543_003",
    "00025543_004",
    "00025543_005",
    "00025543_006",
    "00025543_007",
    "00025543_008",
    "00025543_014",
    "00025543_015",
    "00025543_016",
    "00025543_017",
    "00025543_018",
    "00025543_019",
    "00025554_000",
    "00025566_000",
    "00025567_000",
    "00025577_000",
    "00025581_000",
    "00025583_009",
    "00025585_000",
    "00025587_000",
    "00025589_003",
    "00025596_001",
    "00025600_002",
    "00025600_003",
    "00025603_003",
    "00025621_000",
    "00025621_006",
    "00025621_007",
    "00025621_008",
    "00025621_009",
    "00025623_001",
    "00025623_002",
    "00025635_002",
    "00025635_003",
    "00025635_004",
    "00025635_005",
    "00025635_006",
    "00025635_007",
    "00025635_008",
    "00025645_004",
    "00025662_005",
    "00025662_007",
    "00025662_008",
    "00025666_000",
    "00025674_000",
    "00025680_001",
    "00025680_002",
    "00025680_003",
    "00025680_004",
    "00025680_005",
    "00025680_006",
    "00025681_000",
    "00025681_001",
    "00025683_000",
    "00025683_001",
    "00025683_002",
    "00025683_003",
    "00025685_001",
    "00025688_000",
    "00025688_001",
    "00025688_002",
    "00025690_000",
    "00025692_000",
    "00025697_000",
    "00025697_001",
    "00025698_000",
    "00025707_000",
    "00025707_001",
    "00025707_002",
    "00025707_003",
    "00025707_004",
    "00025707_008",
    "00025707_009",
    "00025707_010",
    "00025707_011",
    "00025707_012",
    "00025707_014",
    "00025711_000",
    "00025711_001",
    "00025720_001",
    "00025732_000",
    "00025733_002",
    "00025733_003",
    "00025733_004",
    "00025733_005",
    "00025733_006",
    "00025733_007",
    "00025733_008",
    "00025735_000",
    "00025735_001",
    "00025735_002",
    "00025737_000",
    "00025745_000",
    "00025745_001",
    "00025745_002",
    "00025745_003",
    "00025750_002",
    "00025750_003",
    "00025754_015",
    "00025761_000",
    "00025761_001",
    "00025761_002",
    "00025761_003",
    "00025761_011",
    "00025763_001",
    "00025766_000",
    "00025768_000",
    "00025768_005",
    "00025769_001",
    "00025769_006",
    "00025769_007",
    "00025769_008",
    "00025781_002",
    "00025781_003",
    "00025793_000",
    "00025799_000",
    "00025809_003",
    "00025812_007",
    "00025823_001",
    "00025832_001",
    "00025832_002",
    "00025832_003",
    "00025832_004",
    "00025833_000",
    "00025844_000",
    "00025846_001",
    "00025851_000",
    "00025851_001",
    "00025851_002",
    "00025851_003",
    "00025851_004",
    "00025851_005",
    "00025867_000",
    "00025877_009",
    "00025877_010",
    "00025877_011",
    "00025877_012",
    "00025884_000",
    "00025884_001",
    "00025884_002",
    "00025890_002",
    "00025895_000",
    "00025897_002",
    "00025897_003",
    "00025897_004",
    "00025903_000",
    "00025904_000",
    "00025904_001",
    "00025904_002",
    "00025904_003",
    "00025905_003",
    "00025908_000",
    "00025909_002",
    "00025909_003",
    "00025909_004",
    "00025913_000",
    "00025925_001",
    "00025925_002",
    "00025925_003",
    "00025936_000",
    "00025936_001",
    "00025936_002",
    "00025936_003",
    "00025936_004",
    "00025936_005",
    "00025936_006",
    "00025954_000",
    "00025954_012",
    "00025954_013",
    "00025954_014",
    "00025954_023",
    "00025954_025",
    "00025954_026",
    "00025954_027",
    "00025954_028",
    "00025954_031",
    "00025954_032",
    "00025954_035",
    "00025954_036",
    "00025954_037",
    "00025954_038",
    "00025954_039",
    "00025954_043",
    "00025959_001",
    "00025960_003",
    "00025964_000",
    "00025964_001",
    "00025964_008",
    "00025977_003",
    "00025994_000",
    "00025994_002",
    "00025997_000",
    "00025997_001",
    "00026009_000",
    "00026011_000",
    "00026019_001",
    "00026021_000",
    "00026023_000",
    "00026024_000",
    "00026024_001",
    "00026024_002",
    "00026024_006",
    "00026024_027",
    "00026031_000",
    "00026031_001",
    "00026039_000",
    "00026041_002",
    "00026043_000",
    "00026059_003",
    "00026069_000",
    "00026072_002",
    "00026072_004",
    "00026072_005",
    "00026072_008",
    "00026072_009",
    "00026072_010",
    "00026072_011",
    "00026072_012",
    "00026072_013",
    "00026072_014",
    "00026072_015",
    "00026072_016",
    "00026072_017",
    "00026072_018",
    "00026072_019",
    "00026088_000",
    "00026092_004",
    "00026095_000",
    "00026098_000",
    "00026098_001",
    "00026098_002",
    "00026098_003",
    "00026098_004",
    "00026098_005",
    "00026098_006",
    "00026098_007",
    "00026098_008",
    "00026098_009",
    "00026098_010",
    "00026098_017",
    "00026098_018",
    "00026098_027",
    "00026102_000",
    "00026112_001",
    "00026112_002",
    "00026112_003",
    "00026120_001",
    "00026132_014",
    "00026134_000",
    "00026134_001",
    "00026134_002",
    "00026136_000",
    "00026136_002",
    "00026136_003",
    "00026143_000",
    "00026143_001",
    "00026151_000",
    "00026156_002",
    "00026156_003",
    "00026164_000",
    "00026166_002",
    "00026166_003",
    "00026169_000",
    "00026176_000",
    "00026185_000",
    "00026185_002",
    "00026185_003",
    "00026185_004",
    "00026185_005",
    "00026185_006",
    "00026187_000",
    "00026191_000",
    "00026202_007",
    "00026203_000",
    "00026203_001",
    "00026203_011",
    "00026208_000",
    "00026221_000",
    "00026221_001",
    "00026221_002",
    "00026221_003",
    "00026221_004",
    "00026221_005",
    "00026221_006",
    "00026221_007",
    "00026221_008",
    "00026221_009",
    "00026221_010",
    "00026221_011",
    "00026221_012",
    "00026221_013",
    "00026221_014",
    "00026221_015",
    "00026221_016",
    "00026223_001",
    "00026223_002",
    "00026223_003",
    "00026223_004",
    "00026223_005",
    "00026224_000",
    "00026232_000",
    "00026235_000",
    "00026236_016",
    "00026236_017",
    "00026236_018",
    "00026236_019",
    "00026236_020",
    "00026236_021",
    "00026239_000",
    "00026244_000",
    "00026244_001",
    "00026244_002",
    "00026247_000",
    "00026261_000",
    "00026261_001",
    "00026261_002",
    "00026261_003",
    "00026261_004",
    "00026261_008",
    "00026263_002",
    "00026263_003",
    "00026263_004",
    "00026263_005",
    "00026263_006",
    "00026263_007",
    "00026263_008",
    "00026263_009",
    "00026263_010",
    "00026263_011",
    "00026263_012",
    "00026263_013",
    "00026263_014",
    "00026275_008",
    "00026282_003",
    "00026282_004",
    "00026283_001",
    "00026301_000",
    "00026301_001",
    "00026301_002",
    "00026301_003",
    "00026306_000",
    "00026315_000",
    "00026315_001",
    "00026318_002",
    "00026325_002",
    "00026325_003",
    "00026325_004",
    "00026337_004",
    "00026337_005",
    "00026337_006",
    "00026337_007",
    "00026338_000",
    "00026341_000",
    "00026341_001",
    "00026341_002",
    "00026341_003",
    "00026341_004",
    "00026341_005",
    "00026341_006",
    "00026341_007",
    "00026341_008",
    "00026349_000",
    "00026349_001",
    "00026352_000",
    "00026358_000",
    "00026360_000",
    "00026363_002",
    "00026363_003",
    "00026365_000",
    "00026372_000",
    "00026372_001",
    "00026383_000",
    "00026387_000",
    "00026387_001",
    "00026387_002",
    "00026392_000",
    "00026392_001",
    "00026392_002",
    "00026392_003",
    "00026392_004",
    "00026392_005",
    "00026410_000",
    "00026412_001",
    "00026412_003",
    "00026412_004",
    "00026412_006",
    "00026412_007",
    "00026412_008",
    "00026421_002",
    "00026421_003",
    "00026421_004",
    "00026439_000",
    "00026443_000",
    "00026443_001",
    "00026443_002",
    "00026444_001",
    "00026450_000",
    "00026473_000",
    "00026474_000",
    "00026480_000",
    "00026480_001",
    "00026480_002",
    "00026480_003",
    "00026480_004",
    "00026480_005",
    "00026480_006",
    "00026480_007",
    "00026480_008",
    "00026484_000",
    "00026508_001",
    "00026515_005",
    "00026515_006",
    "00026516_004",
    "00026516_005",
    "00026521_000",
    "00026531_000",
    "00026543_002",
    "00026546_000",
    "00026546_001",
    "00026546_002",
    "00026546_003",
    "00026546_004",
    "00026546_005",
    "00026546_006",
    "00026546_007",
    "00026546_008",
    "00026546_009",
    "00026548_000",
    "00026548_001",
    "00026557_001",
    "00026561_002",
    "00026565_000",
    "00026567_000",
    "00026568_000",
    "00026574_000",
    "00026577_000",
    "00026578_000",
    "00026587_001",
    "00026607_000",
    "00026608_000",
    "00026612_000",
    "00026622_000",
    "00026622_001",
    "00026622_002",
    "00026626_000",
    "00026637_000",
    "00026638_002",
    "00026652_000",
    "00026652_001",
    "00026652_002",
    "00026656_000",
    "00026683_000",
    "00026683_001",
    "00026683_002",
    "00026683_003",
    "00026694_006",
    "00026713_003",
    "00026714_002",
    "00026715_001",
    "00026717_000",
    "00026723_000",
    "00026737_001",
    "00026738_000",
    "00026738_001",
    "00026746_000",
    "00026746_001",
    "00026753_000",
    "00026768_000",
    "00026768_002",
    "00026769_004",
    "00026771_000",
    "00026775_000",
    "00026780_000",
    "00026798_002",
    "00026820_000",
    "00026820_003",
    "00026822_000",
    "00026828_000",
    "00026831_003",
    "00026831_005",
    "00026833_002",
    "00026835_004",
    "00026837_000",
    "00026849_002",
    "00026855_000",
    "00026855_002",
    "00026860_003",
    "00026863_005",
    "00026865_001",
    "00026875_000",
    "00026876_001",
    "00026876_002",
    "00026909_001",
    "00026910_000",
    "00026911_005",
    "00026919_000",
    "00026938_000",
    "00026940_002",
    "00026956_000",
    "00026956_003",
    "00026958_000",
    "00026964_000",
    "00026968_000",
    "00026969_002",
    "00026969_004",
    "00026982_000",
    "00026982_001",
    "00026993_004",
    "00026993_005",
    "00026993_006",
    "00026996_000",
    "00026999_000",
    "00027020_000",
    "00027020_001",
    "00027023_001",
    "00027023_003",
    "00027028_002",
    "00027028_005",
    "00027038_000",
    "00027038_001",
    "00027056_000",
    "00027064_000",
    "00027077_000",
    "00027079_000",
    "00027081_000",
    "00027097_000",
    "00027097_001",
    "00027107_000",
    "00027107_003",
    "00027107_005",
    "00027117_000",
    "00027136_000",
    "00027172_000",
    "00027177_000",
    "00027189_000",
    "00027189_003",
    "00027194_000",
    "00027195_000",
    "00027196_000",
    "00027199_000",
    "00027199_008",
    "00027201_000",
    "00027205_000",
    "00027206_000",
    "00027207_000",
    "00027238_000",
    "00027240_000",
    "00027267_000",
    "00027282_002",
    "00027305_000",
    "00027310_000",
    "00027311_000",
    "00027311_001",
    "00027311_002",
    "00027311_003",
    "00027311_004",
    "00027320_000",
    "00027322_001",
    "00027322_006",
    "00027332_000",
    "00027342_000",
    "00027343_003",
    "00027343_004",
    "00027343_005",
    "00027365_000",
    "00027366_000",
    "00027370_000",
    "00027371_000",
    "00027378_000",
    "00027378_001",
    "00027380_000",
    "00027381_000",
    "00027383_000",
    "00027385_000",
    "00027386_002",
    "00027387_000",
    "00027388_001",
    "00027394_000",
    "00027397_000",
    "00027403_000",
    "00027424_000",
    "00027433_000",
    "00027433_001",
    "00027457_000",
    "00027500_000",
    "00027507_000",
    "00027509_000",
    "00027514_001",
    "00027528_000",
    "00027537_000",
    "00027553_000",
    "00027560_000",
    "00027564_000",
    "00027580_000",
    "00027590_000",
    "00027594_000",
    "00027613_000",
    "00027652_000",
    "00027675_000",
    "00027702_000",
    "00027725_003",
    "00027725_004",
    "00027726_007",
    "00027728_004",
    "00027728_005",
    "00027742_001",
    "00027757_000",
    "00027759_000",
    "00027761_000"]  # Substitua isso pelos IDs que deseja remover

    # Filtrar o DataFrame para remover as linhas com IDs na lista 'ids_a_remover'
    df = df[~df['StudyInstanceUID'].isin(ids_a_remover)]

    # Salvar o DataFrame atualizado de volta para o arquivo CSV
    df.to_csv('train.csv', index=False)

# Adicionar novas linhas csv
import csv
import os
def adicionar_novas_linas_csv():
    
    # Diretório onde estão as imagens
    diretorio = 'semtubos'

    # Lista para armazenar os nomes dos arquivos sem extensão
    nomes_arquivos_sem_extensao = []

    # Iterar sobre os arquivos no diretório
    for nome_arquivo in os.listdir(diretorio):
        # Verificar se o arquivo é uma imagem PNG
        if nome_arquivo.lower().endswith('.png'):
            # Remover a extensão .png e adicionar o nome do arquivo à lista
            nome_sem_extensao = os.path.splitext(nome_arquivo)[0]
            nomes_arquivos_sem_extensao.append(nome_sem_extensao)

    # Abra o arquivo CSV em modo de escrita e defina o delimitador como vírgula (',')
    with open('trainteste.csv', 'a', newline='') as arquivo_csv:
        escritor_csv = csv.writer(arquivo_csv)

        # Loop sobre os IDs e adicionar nova linha para cada um
        for id_novo in nomes_arquivos_sem_extensao:
            # Criar uma lista representando a nova linha
            nova_linha = [id_novo] + [0]*11 + ['unknown']

            # Escrever a nova linha no arquivo CSV
            escritor_csv.writerow(nova_linha)

import random

def gerar_csv_classi_binario(pathValidationCsv, nameCsv):
    
    lista_mascaras_ids = []
    
    lista_ids_sem_mascara = []
    with open(pathValidationCsv, "r") as arquivo_csv:
        read_csv = csv.reader(arquivo_csv)
        temp_semtubo = 0
        temp_tubo = 0
        for i in read_csv:
            if i[0] != 'ID':
                if 'Sem - Tubo' in i[1]:
                    lista_ids_sem_mascara.append(i[0])
                    temp_semtubo += 1
                else:
                    lista_mascaras_ids.append(i[0])
                    temp_tubo += 1
        print(f"Imagens com tubo: {temp_tubo}")
        print(f"Imagens sem tubo: {temp_semtubo}")
    
    with open(os.path.join("/content/drive/MyDrive/Colab Notebooks/CatheterClassify/data/labels", nameCsv), "w", newline='') as arquivo_csv:
        write_csv = csv.writer(arquivo_csv)
        
        write_csv.writerow(['ID', 'Conteudo', 'Predict'])
        for k in lista_mascaras_ids:
            write_csv.writerow([k, 1, 0])

        for k in lista_ids_sem_mascara:
            write_csv.writerow([k, 0, 0])

def addingMoreDataNoTube():
    def addingCsv(list_ids, path='Máscara de validação (1) copy.csv'):
        with open(path, 'a', newline='') as folder_csv:
            write_csv = csv.writer(folder_csv)
            path_arquivo = "/content/xrays/train_imagens/PreProcessing/" # "/content/drive/MyDrive/Colab Notebooks/CatheterClassify/data/raw/dataset/xrays/train/"  #
            path_mask = "/content/drive/MyDrive/Colab Notebooks/CatheterClassify/data/raw/dataset/masks/NGT/"  #"/content/mask_imagens/" #
        
            # Loop sobre os IDs e adicionar nova linha para cada um
            for id_novo in list_ids:
                write_csv.writerow([id_novo, ['Sem - Tubo'], path_arquivo+id_novo+".jpg", path_mask+"semtubo.jpg"])
                
    def withOutTubeDataCSV(path_csv):
        list_ids = []
        with open(path_csv, "r") as folder_csv:
            read_csv = csv.DictReader(folder_csv)
            for row in read_csv:
                if "Sem - Tubo" in row['labels']:
                    list_ids.append(row["ID"])
        return list_ids
    def images_sem_tubo(path_csv):
        
        tube_position1 = 'NGT - Normal'
        tube_position2 = 'NGT - Borderline'
        tube_position3 = 'NGT - Abnormal'
        tube_position4 = 'NGT - Incompletely Imaged'
        
        with open(path_csv,'r') as folder_csv:
            read_csv = csv.DictReader(folder_csv)
            list_without_tubes_one = []
            for line in read_csv:
                if line[tube_position1] == '0' and line[tube_position2] == '0' and line[tube_position3] == '0' and line[tube_position4] == '0':
                    if line['PatientID'] != 'unknown':
                        list_without_tubes_one.append(line['StudyInstanceUID'])
            return list_without_tubes_one
            
    list_ids_on_csv = []
    list_ids_on_csv.extend(withOutTubeDataCSV("NGT labels train mask.csv"))
    list_ids_on_csv.extend(withOutTubeDataCSV("Máscara de validação (1).csv"))
    list_ids_on_csv.extend(withOutTubeDataCSV("NGT labels.csv"))
    print(len(list_ids_on_csv))
    
    list_ids_noTube = []
    list_ids_noTube.extend(images_sem_tubo("labels train.csv"))
    
    for i in list_ids_on_csv:
        list_ids_noTube.remove(i)
    
    list_new_data_add = []
    
    for i in range(int(len(list_ids_on_csv)*0.35)):
        if i % 10 == 0:
            random.shuffle(list_ids_noTube)
        list_new_data_add.append(list_ids_noTube.pop())
    
    print(len(list_new_data_add))
    
    percentage_balancing = int(0.6 *len(list_new_data_add))
    print(percentage_balancing)
    random.shuffle(list_new_data_add)
    
    addingCsv(list_new_data_add[:percentage_balancing], "train_mask.csv")
    csv_divide = list_new_data_add[percentage_balancing:]
    random.shuffle(csv_divide)
    addingCsv(csv_divide[int(len(csv_divide)*0.5):], "test_mask.csv")
    addingCsv(csv_divide[:int(len(csv_divide)*0.5)], "validation_mask.csv")
    
addingMoreDataNoTube()
